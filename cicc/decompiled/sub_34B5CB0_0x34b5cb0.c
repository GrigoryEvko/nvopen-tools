// Function: sub_34B5CB0
// Address: 0x34b5cb0
//
unsigned __int64 __fastcall sub_34B5CB0(unsigned int *a1, int a2, unsigned __int64 *a3, __int64 a4)
{
  unsigned __int64 result; // rax
  unsigned int v7; // r13d
  int v8; // edx
  char *v9; // rsi
  unsigned int v11[13]; // [rsp+1Ch] [rbp-34h] BYREF

  result = *a1;
  if ( (_DWORD)result )
  {
    v7 = 0;
    do
    {
      LODWORD(result) = *(_DWORD *)(*((_QWORD *)a1 + 4) + 4LL * v7);
      do
      {
        v8 = result;
        result = *(unsigned int *)(*((_QWORD *)a1 + 1) + 4LL * (unsigned int)result);
      }
      while ( v8 != (_DWORD)result );
      if ( a2 == (_DWORD)result )
      {
        v11[0] = v7;
        result = sub_34B40C0(a4, v11);
        if ( result )
        {
          v9 = (char *)a3[1];
          if ( v9 == (char *)a3[2] )
          {
            result = sub_34B5B30(a3, v9, v11);
          }
          else
          {
            if ( v9 )
            {
              *(_DWORD *)v9 = v7;
              v9 = (char *)a3[1];
            }
            result = (unsigned __int64)a3;
            a3[1] = (unsigned __int64)(v9 + 4);
          }
        }
      }
      ++v7;
    }
    while ( *a1 != v7 );
  }
  return result;
}
