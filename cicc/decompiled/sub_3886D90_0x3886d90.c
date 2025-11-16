// Function: sub_3886D90
// Address: 0x3886d90
//
__int64 __fastcall sub_3886D90(unsigned __int8 **a1, unsigned int a2, unsigned int a3)
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
  signed __int64 v14; // rdx
  unsigned __int8 *v15; // r13
  char *v16; // rax
  _QWORD v17[2]; // [rsp+0h] [rbp-40h] BYREF
  char v18; // [rsp+10h] [rbp-30h]
  char v19; // [rsp+11h] [rbp-2Fh]
  __int64 savedregs; // [rsp+40h] [rbp+0h] BYREF

  if ( **a1 == 34 )
  {
    ++*a1;
    do
    {
      v9 = sub_3880F40(a1);
      if ( v9 == -1 )
      {
        v19 = 1;
        v10 = "end of file in global variable name";
LABEL_6:
        v11 = (unsigned __int64)a1[6];
        v17[0] = v10;
        v18 = 3;
        sub_38814C0((__int64)a1, v11, (__int64)v17);
        return 1;
      }
    }
    while ( v9 != 34 );
    sub_2241130(
      (unsigned __int64 *)a1 + 8,
      0,
      (unsigned __int64)a1[9],
      a1[6] + 2,
      (size_t)&(*a1)[~(unsigned __int64)(a1[6] + 2)]);
    sub_3880B30((unsigned __int64 *)a1 + 8);
    v14 = (signed __int64)a1[9];
    if ( v14 )
    {
      v15 = a1[8];
      if ( v14 < 0 )
        v14 = 0x7FFFFFFFFFFFFFFFLL;
      v16 = (char *)memchr(a1[8], 0, v14);
      if ( v16 )
      {
        if ( v16 - (char *)v15 != -1 )
        {
          v19 = 1;
          v10 = "Null bytes are not allowed in names";
          goto LABEL_6;
        }
      }
    }
    return a2;
  }
  if ( (unsigned __int8)sub_3880FB0((unsigned __int64 *)a1) )
    return a2;
  v3 = *a1;
  if ( (unsigned int)**a1 - 48 > 9 )
    return 1;
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
  v7 = sub_3881F70((__int64)a1, (char *)a1[6] + 1, v4);
  if ( v7 != (unsigned int)v7 )
  {
    v8 = (unsigned __int64)a1[6];
    v19 = 1;
    v18 = 3;
    v17[0] = "invalid value number (too large)!";
    sub_38814C0((__int64)a1, v8, (__int64)v17);
  }
  *((_DWORD *)a1 + 24) = v7;
  return v5;
}
