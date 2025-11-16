// Function: sub_1321770
// Address: 0x1321770
//
__int64 __fastcall sub_1321770(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 *a5, __int64 a6, __int64 a7)
{
  __int64 result; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int64 v12; // r8
  char *v13; // rbx
  unsigned int v14; // eax
  __int64 v15; // rdi
  _QWORD v16[5]; // [rsp+8h] [rbp-28h] BYREF

  result = 1;
  if ( !(a7 | a6) )
  {
    sub_134B260();
    v10 = sub_134B2C0(a1);
    v16[0] = v10;
    if ( a4 && a5 )
    {
      v11 = *a5;
      if ( *a5 == 8 )
      {
        *a4 = v10;
        return 0;
      }
      else
      {
        if ( (unsigned __int64)*a5 > 8 )
          v11 = 8;
        if ( (unsigned int)v11 >= 8 )
        {
          *a4 = v10;
          v12 = (unsigned __int64)(a4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          *(__int64 *)((char *)a4 + (unsigned int)v11 - 8) = *(_QWORD *)((char *)&v16[-1] + (unsigned int)v11);
          v13 = (char *)a4 - v12;
          if ( (((_DWORD)v11 + (_DWORD)v13) & 0xFFFFFFF8) >= 8 )
          {
            v14 = 0;
            do
            {
              v15 = v14;
              v14 += 8;
              *(_QWORD *)(v12 + v15) = *(_QWORD *)((char *)v16 - v13 + v15);
            }
            while ( v14 < (((_DWORD)v11 + (_DWORD)v13) & 0xFFFFFFF8) );
          }
        }
        else if ( (v11 & 4) != 0 )
        {
          *(_DWORD *)a4 = v16[0];
          *(_DWORD *)((char *)a4 + (unsigned int)v11 - 4) = *(_DWORD *)((char *)v16 + (unsigned int)v11 - 4);
        }
        else if ( (_DWORD)v11 )
        {
          *(_BYTE *)a4 = v16[0];
          if ( (v11 & 2) != 0 )
            *(_WORD *)((char *)a4 + (unsigned int)v11 - 2) = *(_WORD *)((char *)v16 + (unsigned int)v11 - 2);
        }
        *a5 = v11;
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
