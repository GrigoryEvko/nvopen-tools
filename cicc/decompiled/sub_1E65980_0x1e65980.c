// Function: sub_1E65980
// Address: 0x1e65980
//
_QWORD *__fastcall sub_1E65980(_QWORD *a1, __int64 *a2)
{
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 *v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 *v14; // rsi
  __int64 *v15; // rcx
  unsigned __int64 v16; // [rsp+0h] [rbp-140h] BYREF
  char v17; // [rsp+10h] [rbp-130h]
  __int64 v18; // [rsp+20h] [rbp-120h] BYREF
  __int64 *v19; // [rsp+28h] [rbp-118h]
  __int64 *v20; // [rsp+30h] [rbp-110h]
  __int64 v21; // [rsp+38h] [rbp-108h]
  int v22; // [rsp+40h] [rbp-100h]
  _QWORD v23[8]; // [rsp+48h] [rbp-F8h] BYREF
  unsigned __int64 v24; // [rsp+88h] [rbp-B8h] BYREF
  __int64 v25; // [rsp+90h] [rbp-B0h]
  __int64 v26; // [rsp+98h] [rbp-A8h]
  _QWORD v27[20]; // [rsp+A0h] [rbp-A0h] BYREF

  v3 = a2[4];
  v19 = v23;
  memset(v27, 0, 0x80u);
  v27[1] = &v27[5];
  v27[2] = &v27[5];
  v4 = *a2;
  LODWORD(v27[3]) = 8;
  v20 = v23;
  v23[0] = v4 & 0xFFFFFFFFFFFFFFF8LL;
  v16 = v4 & 0xFFFFFFFFFFFFFFF8LL;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v21 = 0x100000008LL;
  v22 = 0;
  v18 = 1;
  v17 = 0;
  sub_1E65930(&v24, (__int64)&v16);
  v5 = v19;
  if ( v20 != v19 )
    goto LABEL_2;
  v14 = &v19[HIDWORD(v21)];
  if ( v19 == v14 )
  {
LABEL_21:
    if ( HIDWORD(v21) >= (unsigned int)v21 )
    {
LABEL_2:
      sub_16CCBA0((__int64)&v18, v3);
      goto LABEL_3;
    }
    ++HIDWORD(v21);
    *v14 = v3;
    ++v18;
  }
  else
  {
    v15 = 0;
    while ( v3 != *v5 )
    {
      if ( *v5 == -2 )
        v15 = v5;
      if ( v14 == ++v5 )
      {
        if ( !v15 )
          goto LABEL_21;
        *v15 = v3;
        --v22;
        ++v18;
        break;
      }
    }
  }
LABEL_3:
  sub_16CCEE0(a1, (__int64)(a1 + 5), 8, (__int64)&v18);
  v6 = v24;
  v24 = 0;
  a1[13] = v6;
  v7 = v25;
  v25 = 0;
  a1[14] = v7;
  v8 = v26;
  v26 = 0;
  a1[15] = v8;
  sub_16CCEE0(a1 + 16, (__int64)(a1 + 21), 8, (__int64)v27);
  v9 = v27[13];
  v10 = v24;
  v27[13] = 0;
  a1[29] = v9;
  v11 = v27[14];
  v27[14] = 0;
  a1[30] = v11;
  v12 = v27[15];
  v27[15] = 0;
  a1[31] = v12;
  if ( v10 )
    j_j___libc_free_0(v10, v26 - v10);
  if ( v20 != v19 )
    _libc_free((unsigned __int64)v20);
  if ( v27[13] )
    j_j___libc_free_0(v27[13], v27[15] - v27[13]);
  if ( v27[2] != v27[1] )
    _libc_free(v27[2]);
  return a1;
}
