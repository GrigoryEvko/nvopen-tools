// Function: sub_11FF150
// Address: 0x11ff150
//
__int64 __fastcall sub_11FF150(unsigned __int8 **a1, unsigned int a2, unsigned int a3)
{
  unsigned __int8 *v3; // rax
  char *v4; // r8
  unsigned int v5; // ebx
  char *v6; // rax
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // rsi
  int v9; // eax
  const char *v10; // rax
  unsigned __int64 v11; // rsi
  __int64 result; // rax
  char v14; // r8
  size_t v15; // rdx
  _QWORD v16[4]; // [rsp+0h] [rbp-50h] BYREF
  char v17; // [rsp+20h] [rbp-30h]
  char v18; // [rsp+21h] [rbp-2Fh]
  __int64 savedregs; // [rsp+50h] [rbp+0h] BYREF

  if ( **a1 == 34 )
  {
    ++*a1;
    do
    {
      v9 = sub_11FD3B0(a1);
      if ( v9 == -1 )
      {
        v18 = 1;
        v10 = "end of file in global variable name";
LABEL_6:
        v11 = (unsigned __int64)a1[7];
        v16[0] = v10;
        v17 = 3;
        sub_11FD800((__int64)a1, v11, (__int64)v16, 2);
        return 1;
      }
    }
    while ( v9 != 34 );
    sub_2241130(a1 + 9, 0, a1[10], a1[7] + 2, &(*a1)[~(unsigned __int64)(a1[7] + 2)]);
    sub_11FCF00(a1 + 9);
    v15 = (size_t)a1[10];
    if ( v15 && memchr(a1[9], 0, v15) )
    {
      v18 = 1;
      v10 = "NUL character is not allowed in names";
      goto LABEL_6;
    }
    return a2;
  }
  else
  {
    v14 = sub_11FD420(a1);
    result = a2;
    if ( !v14 )
    {
      v3 = *a1;
      if ( (unsigned int)**a1 - 48 > 9 )
      {
        return 1;
      }
      else
      {
        savedregs = (__int64)&savedregs;
        v4 = (char *)(v3 + 1);
        v5 = a3;
        *a1 = v3 + 1;
        if ( (unsigned int)v3[1] - 48 <= 9 )
        {
          v6 = (char *)(v3 + 2);
          do
          {
            v4 = v6;
            *a1 = (unsigned __int8 *)v6++;
          }
          while ( (unsigned int)(unsigned __int8)*v4 - 48 <= 9 );
        }
        v7 = sub_11FE300((__int64)a1, (char *)a1[7] + 1, v4);
        if ( v7 != (unsigned int)v7 )
        {
          v8 = (unsigned __int64)a1[7];
          v18 = 1;
          v17 = 3;
          v16[0] = "invalid value number (too large)";
          sub_11FD800((__int64)a1, v8, (__int64)v16, 2);
        }
        *((_DWORD *)a1 + 26) = v7;
        return v5;
      }
    }
  }
  return result;
}
