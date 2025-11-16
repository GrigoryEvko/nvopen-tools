// Function: sub_255B9A0
// Address: 0x255b9a0
//
_BOOL8 __fastcall sub_255B9A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _BOOL4 v3; // r12d
  __int64 v5; // [rsp+18h] [rbp-158h] BYREF
  _QWORD v6[4]; // [rsp+20h] [rbp-150h] BYREF
  void *v7; // [rsp+40h] [rbp-130h] BYREF
  int v8; // [rsp+48h] [rbp-128h]
  const void *v9; // [rsp+50h] [rbp-120h] BYREF
  unsigned int v10; // [rsp+58h] [rbp-118h]
  const void *v11; // [rsp+60h] [rbp-110h] BYREF
  unsigned int v12; // [rsp+68h] [rbp-108h]
  const void *v13; // [rsp+70h] [rbp-100h] BYREF
  unsigned int v14; // [rsp+78h] [rbp-F8h]
  const void *v15; // [rsp+80h] [rbp-F0h] BYREF
  unsigned int v16; // [rsp+88h] [rbp-E8h]
  void *v17; // [rsp+90h] [rbp-E0h] BYREF
  int v18; // [rsp+98h] [rbp-D8h]
  const void *v19; // [rsp+A0h] [rbp-D0h] BYREF
  unsigned int v20; // [rsp+A8h] [rbp-C8h]
  const void *v21; // [rsp+B0h] [rbp-C0h] BYREF
  unsigned int v22; // [rsp+B8h] [rbp-B8h]
  const void *v23; // [rsp+C0h] [rbp-B0h] BYREF
  unsigned int v24; // [rsp+C8h] [rbp-A8h]
  const void *v25; // [rsp+D0h] [rbp-A0h] BYREF
  unsigned int v26; // [rsp+D8h] [rbp-98h]
  const void *v27[18]; // [rsp+E0h] [rbp-90h] BYREF

  sub_AADB10((__int64)v27, *(_DWORD *)(a1 + 96), 0);
  v7 = &unk_4A16D38;
  v8 = (int)v27[1];
  v10 = (unsigned int)v27[1];
  if ( LODWORD(v27[1]) > 0x40 )
    sub_C43780((__int64)&v9, v27);
  else
    v9 = v27[0];
  v12 = (unsigned int)v27[3];
  if ( LODWORD(v27[3]) > 0x40 )
    sub_C43780((__int64)&v11, &v27[2]);
  else
    v11 = v27[2];
  sub_AADB10((__int64)&v13, (unsigned int)v27[1], 1);
  if ( LODWORD(v27[3]) > 0x40 && v27[2] )
    j_j___libc_free_0_0((unsigned __int64)v27[2]);
  if ( LODWORD(v27[1]) > 0x40 && v27[0] )
    j_j___libc_free_0_0((unsigned __int64)v27[0]);
  v2 = *(_QWORD *)(a1 + 80);
  v6[1] = a2;
  v5 = v2;
  memset(v27, 0, 0x58u);
  v6[0] = &v5;
  v6[2] = a1;
  v6[3] = v27;
  if ( (unsigned __int8)sub_2527330(
                          a2,
                          (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_258A7E0,
                          (__int64)v6,
                          a1,
                          1u,
                          1u) )
  {
    if ( !LOBYTE(v27[10]) )
      goto LABEL_20;
    v18 = (int)v27[3];
    if ( LODWORD(v27[3]) > 0x40 )
      sub_C43780((__int64)&v17, &v27[2]);
    else
      v17 = (void *)v27[2];
    v20 = (unsigned int)v27[5];
    if ( LODWORD(v27[5]) > 0x40 )
      sub_C43780((__int64)&v19, &v27[4]);
    else
      v19 = v27[4];
    sub_254F7F0((__int64)&v7, (__int64)&v17);
    sub_969240((__int64 *)&v19);
    sub_969240((__int64 *)&v17);
    v17 = &unk_4A16D38;
    v18 = v8;
    v20 = v10;
    if ( v10 > 0x40 )
      sub_C43780((__int64)&v19, &v9);
    else
      v19 = v9;
    v22 = v12;
    if ( v12 > 0x40 )
      sub_C43780((__int64)&v21, &v11);
    else
      v21 = v11;
    v24 = v14;
    if ( v14 > 0x40 )
      sub_C43780((__int64)&v23, &v13);
    else
      v23 = v13;
    v26 = v16;
    if ( v16 > 0x40 )
      sub_C43780((__int64)&v25, &v15);
    else
      v25 = v15;
    sub_253FFA0((__int64)&v17);
  }
  else
  {
    if ( v10 <= 0x40 && v14 <= 0x40 )
    {
      v10 = v14;
      v9 = v13;
    }
    else
    {
      sub_C43990((__int64)&v9, (__int64)&v13);
    }
    if ( v12 <= 0x40 && v16 <= 0x40 )
    {
      v12 = v16;
      v11 = v15;
    }
    else
    {
      sub_C43990((__int64)&v11, (__int64)&v15);
    }
  }
  if ( LOBYTE(v27[10]) )
  {
    LOBYTE(v27[10]) = 0;
    sub_253FFA0((__int64)v27);
  }
LABEL_20:
  v3 = sub_255B670(a1 + 88, (__int64)&v7);
  v7 = &unk_4A16D38;
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0((unsigned __int64)v15);
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0((unsigned __int64)v13);
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0((unsigned __int64)v11);
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0((unsigned __int64)v9);
  return v3;
}
