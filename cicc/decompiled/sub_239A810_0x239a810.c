// Function: sub_239A810
// Address: 0x239a810
//
_QWORD *__fastcall sub_239A810(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  _QWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v11; // rbx
  char *v12; // r14
  char *v13; // r12
  __int64 v14; // rsi
  char *v15; // rbx
  char *v16; // r12
  __int64 v17; // rsi
  unsigned __int64 v19[2]; // [rsp+0h] [rbp-140h] BYREF
  char v20; // [rsp+10h] [rbp-130h] BYREF
  char *v21; // [rsp+38h] [rbp-108h]
  int v22; // [rsp+40h] [rbp-100h]
  char v23; // [rsp+48h] [rbp-F8h] BYREF
  __int64 v24; // [rsp+78h] [rbp-C8h]
  unsigned int v25; // [rsp+88h] [rbp-B8h]
  unsigned __int64 v26[2]; // [rsp+90h] [rbp-B0h] BYREF
  char v27; // [rsp+A0h] [rbp-A0h] BYREF
  char *v28; // [rsp+C8h] [rbp-78h]
  int v29; // [rsp+D0h] [rbp-70h]
  char v30; // [rsp+D8h] [rbp-68h] BYREF
  __int64 v31; // [rsp+108h] [rbp-38h]
  unsigned int v32; // [rsp+118h] [rbp-28h]

  sub_2D423B0(v19, a2 + 8);
  sub_239A530((__int64)v26, (__int64)v19, v2, v3, v4, v5);
  v6 = (_QWORD *)sub_22077B0(0x98u);
  v11 = v6;
  if ( v6 )
  {
    *v6 = &unk_4A0B0B0;
    sub_239A530((__int64)(v6 + 1), (__int64)v26, v7, v8, v9, v10);
  }
  sub_C7D6A0(v31, 16LL * v32, 8);
  v12 = v28;
  v13 = &v28[32 * v29];
  if ( v28 != v13 )
  {
    do
    {
      v14 = *((_QWORD *)v13 - 2);
      v13 -= 32;
      if ( v14 )
        sub_B91220((__int64)(v13 + 16), v14);
    }
    while ( v12 != v13 );
    v13 = v28;
  }
  if ( v13 != &v30 )
    _libc_free((unsigned __int64)v13);
  if ( (char *)v26[0] != &v27 )
    _libc_free(v26[0]);
  *a1 = v11;
  sub_C7D6A0(v24, 16LL * v25, 8);
  v15 = v21;
  v16 = &v21[32 * v22];
  if ( v21 != v16 )
  {
    do
    {
      v17 = *((_QWORD *)v16 - 2);
      v16 -= 32;
      if ( v17 )
        sub_B91220((__int64)(v16 + 16), v17);
    }
    while ( v15 != v16 );
    v16 = v21;
  }
  if ( v16 != &v23 )
    _libc_free((unsigned __int64)v16);
  if ( (char *)v19[0] != &v20 )
    _libc_free(v19[0]);
  return a1;
}
