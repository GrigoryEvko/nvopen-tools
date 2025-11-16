// Function: sub_13207C0
// Address: 0x13207c0
//
__int64 __fastcall sub_13207C0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 *a5, __int64 a6, __int64 a7)
{
  __int64 result; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned __int64 v10; // r9
  char *v11; // rcx
  char *v12; // rsi
  unsigned int v13; // ecx
  unsigned int v14; // ecx
  unsigned int v15; // eax
  __int64 v16; // rdi
  __int64 v17; // [rsp+0h] [rbp-8h] BYREF

  result = 1;
  if ( !(a7 | a6) )
  {
    v8 = qword_4C6F080[3];
    v17 = v8;
    if ( a4 && a5 )
    {
      v9 = *a5;
      if ( *a5 == 8 )
      {
        *a4 = v8;
        return 0;
      }
      else
      {
        if ( (unsigned __int64)*a5 > 8 )
          v9 = 8;
        if ( (unsigned int)v9 >= 8 )
        {
          *a4 = v8;
          v10 = (unsigned __int64)(a4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)((char *)a4 + (unsigned int)v9 - 8) = *(__int64 *)((char *)&v17 + (unsigned int)v9 - 8);
          v11 = (char *)a4 - v10;
          v12 = (char *)((char *)&v17 - v11);
          v13 = (v9 + (_DWORD)v11) & 0xFFFFFFF8;
          if ( v13 >= 8 )
          {
            v14 = v13 & 0xFFFFFFF8;
            v15 = 0;
            do
            {
              v16 = v15;
              v15 += 8;
              *(_QWORD *)(v10 + v16) = *(_QWORD *)&v12[v16];
            }
            while ( v15 < v14 );
          }
        }
        else if ( (v9 & 4) != 0 )
        {
          *(_DWORD *)a4 = v17;
          *(_DWORD *)((char *)a4 + (unsigned int)v9 - 4) = *(_DWORD *)((char *)&v17 + (unsigned int)v9 - 4);
        }
        else if ( (_DWORD)v9 )
        {
          *(_BYTE *)a4 = v17;
          if ( (v9 & 2) != 0 )
            *(_WORD *)((char *)a4 + (unsigned int)v9 - 2) = *(_WORD *)((char *)&v17 + (unsigned int)v9 - 2);
        }
        *a5 = v9;
        return 22;
      }
    }
    else
    {
      return 0;
    }
  }
  return result;
}
