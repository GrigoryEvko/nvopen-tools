// Function: sub_131FA60
// Address: 0x131fa60
//
__int64 __fastcall sub_131FA60(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 *a5, __int64 a6, __int64 a7)
{
  __int64 result; // rax
  __int64 v8; // rdx
  unsigned __int64 v9; // r9
  char *v10; // rcx
  char *v11; // rsi
  unsigned int v12; // ecx
  unsigned int v13; // ecx
  unsigned int v14; // eax
  __int64 v15; // rdi
  __int64 v16; // [rsp+0h] [rbp-8h] BYREF

  result = 1;
  if ( !(a7 | a6) )
  {
    v16 = qword_4C6F230;
    if ( a4 && a5 )
    {
      v8 = *a5;
      if ( *a5 == 8 )
      {
        *a4 = qword_4C6F230;
        return 0;
      }
      else
      {
        if ( (unsigned __int64)*a5 > 8 )
          v8 = 8;
        if ( (unsigned int)v8 >= 8 )
        {
          *a4 = qword_4C6F230;
          v9 = (unsigned __int64)(a4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)((char *)a4 + (unsigned int)v8 - 8) = *(__int64 *)((char *)&v16 + (unsigned int)v8 - 8);
          v10 = (char *)a4 - v9;
          v11 = (char *)((char *)&v16 - v10);
          v12 = (v8 + (_DWORD)v10) & 0xFFFFFFF8;
          if ( v12 >= 8 )
          {
            v13 = v12 & 0xFFFFFFF8;
            v14 = 0;
            do
            {
              v15 = v14;
              v14 += 8;
              *(_QWORD *)(v9 + v15) = *(_QWORD *)&v11[v15];
            }
            while ( v14 < v13 );
          }
        }
        else if ( (v8 & 4) != 0 )
        {
          *(_DWORD *)a4 = v16;
          *(_DWORD *)((char *)a4 + (unsigned int)v8 - 4) = *(_DWORD *)((char *)&v16 + (unsigned int)v8 - 4);
        }
        else if ( (_DWORD)v8 )
        {
          *(_BYTE *)a4 = v16;
          if ( (v8 & 2) != 0 )
            *(_WORD *)((char *)a4 + (unsigned int)v8 - 2) = *(_WORD *)((char *)&v16 + (unsigned int)v8 - 2);
        }
        *a5 = v8;
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
