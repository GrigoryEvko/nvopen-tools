// Function: sub_13212B0
// Address: 0x13212b0
//
__int64 __fastcall sub_13212B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 *a4,
        unsigned __int64 *a5,
        __int64 a6,
        __int64 a7)
{
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdx
  unsigned __int64 v12; // rdi
  char *v13; // r12
  char *v14; // rcx
  unsigned int v15; // r12d
  unsigned int v16; // r12d
  unsigned int v17; // eax
  __int64 v18; // rsi
  _QWORD v19[3]; // [rsp+8h] [rbp-18h] BYREF

  if ( a4 && a5 && *a5 == 8 )
  {
    if ( a6 && a7 == 32 )
    {
      v9 = sub_1308BE0(*(_QWORD *)a6, *(_QWORD *)(a6 + 8), *(_QWORD *)(a6 + 16), *(_DWORD *)(a6 + 24), (int)a5, a6);
      v10 = *a5;
      v19[0] = v9;
      if ( v10 == 8 )
      {
        *a4 = v9;
        return 0;
      }
      else
      {
        if ( v10 > 8 )
          v10 = 8;
        if ( (unsigned int)v10 >= 8 )
        {
          *a4 = v9;
          v12 = (unsigned __int64)(a4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          *(unsigned __int64 *)((char *)a4 + (unsigned int)v10 - 8) = *(_QWORD *)((char *)&v19[-1] + (unsigned int)v10);
          v13 = (char *)a4 - v12;
          v14 = (char *)((char *)v19 - v13);
          v15 = (v10 + (_DWORD)v13) & 0xFFFFFFF8;
          if ( v15 >= 8 )
          {
            v16 = v15 & 0xFFFFFFF8;
            v17 = 0;
            do
            {
              v18 = v17;
              v17 += 8;
              *(_QWORD *)(v12 + v18) = *(_QWORD *)&v14[v18];
            }
            while ( v17 < v16 );
          }
        }
        else if ( (v10 & 4) != 0 )
        {
          *(_DWORD *)a4 = v19[0];
          *(_DWORD *)((char *)a4 + (unsigned int)v10 - 4) = *(_DWORD *)((char *)v19 + (unsigned int)v10 - 4);
        }
        else if ( (_DWORD)v10 )
        {
          *(_BYTE *)a4 = v19[0];
          if ( (v10 & 2) != 0 )
            *(_WORD *)((char *)a4 + (unsigned int)v10 - 2) = *(_WORD *)((char *)v19 + (unsigned int)v10 - 2);
        }
        *a5 = v10;
        return 22;
      }
    }
    else
    {
      return 22;
    }
  }
  else
  {
    *a5 = 0;
    return 22;
  }
}
