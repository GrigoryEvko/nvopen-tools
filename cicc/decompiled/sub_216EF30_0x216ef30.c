// Function: sub_216EF30
// Address: 0x216ef30
//
__int64 __fastcall sub_216EF30(const char *a1, __int64 a2)
{
  unsigned int v3; // eax
  int v4; // edx
  int v5; // ecx
  unsigned __int64 *v6; // rsi
  unsigned __int64 *v7; // rax
  unsigned int v8; // ecx
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  const char **v11; // rdx
  _BYTE *v12; // rsi
  char v13; // cl
  int v14; // edx
  __int64 result; // rax
  unsigned __int64 v16; // [rsp+8h] [rbp-58h] BYREF
  _BYTE v17[16]; // [rsp+10h] [rbp-50h] BYREF
  const char *v18[2]; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v19[6]; // [rsp+30h] [rbp-30h] BYREF

  v16 = 0;
  v3 = strlen(a1);
  if ( v3 > 8 )
  {
    if ( byte_4FD2968 )
    {
      v4 = 7;
      v5 = 8;
    }
    else
    {
      byte_4FD2968 = 1;
      v18[0] = (const char *)v19;
      sub_216EE30((__int64 *)v18, "Function too large, generated debug information may not be accurate.", (__int64)"");
      sub_1C3F040((__int64)v18);
      if ( (_QWORD *)v18[0] != v19 )
        j_j___libc_free_0(v18[0], v19[0] + 1LL);
      v4 = 7;
      v5 = 8;
    }
    goto LABEL_6;
  }
  if ( v3 )
  {
    v5 = v3;
    v4 = v3 - 1;
LABEL_6:
    v6 = (unsigned __int64 *)&v17[v5 - 1 - 7];
    v7 = &v16;
    v8 = v4 + (unsigned int)&v16;
    do
    {
      v9 = v8 - (unsigned int)v7;
      v7 = (unsigned __int64 *)((char *)v7 + 1);
      *((_BYTE *)v7 - 1) = a1[v9];
    }
    while ( v6 != v7 );
    v10 = v16;
    goto LABEL_9;
  }
  v10 = 0;
LABEL_9:
  v11 = (const char **)v17;
  do
  {
    v12 = v11;
    v11 = (const char **)((char *)v11 + 1);
    v13 = v10 & 0x7F;
    v10 >>= 7;
    if ( !v10 )
    {
      *v12 = v13;
      v14 = (_DWORD)v11 - (unsigned int)v17;
      goto LABEL_12;
    }
    *((_BYTE *)v11 - 1) = v13 | 0x80;
  }
  while ( v11 != v18 );
  v14 = 16;
  if ( !byte_4FD2968 )
  {
    byte_4FD2968 = 1;
    v18[0] = (const char *)v19;
    sub_216EE30((__int64 *)v18, "Function too large, generated debug information may not be accurate.", (__int64)"");
    sub_1C3F040((__int64)v18);
    if ( (_QWORD *)v18[0] != v19 )
      j_j___libc_free_0(v18[0], v19[0] + 1LL);
    v14 = 16;
  }
LABEL_12:
  v18[0] = (const char *)v19;
  sub_216EE30((__int64 *)v18, v17, (__int64)&v17[v14]);
  result = sub_2241490(a2, v18[0], v18[1]);
  if ( (_QWORD *)v18[0] != v19 )
    return j_j___libc_free_0(v18[0], v19[0] + 1LL);
  return result;
}
