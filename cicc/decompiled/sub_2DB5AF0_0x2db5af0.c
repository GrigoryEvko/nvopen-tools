// Function: sub_2DB5AF0
// Address: 0x2db5af0
//
__int64 __fastcall sub_2DB5AF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v10; // [rsp+10h] [rbp-4F0h] BYREF
  __int64 *v11; // [rsp+18h] [rbp-4E8h]
  int v12; // [rsp+20h] [rbp-4E0h]
  int v13; // [rsp+24h] [rbp-4DCh]
  int v14; // [rsp+28h] [rbp-4D8h]
  char v15; // [rsp+2Ch] [rbp-4D4h]
  __int64 v16; // [rsp+30h] [rbp-4D0h] BYREF
  __int64 *v17; // [rsp+70h] [rbp-490h]
  __int64 v18; // [rsp+78h] [rbp-488h]
  __int64 v19[24]; // [rsp+80h] [rbp-480h] BYREF
  unsigned __int64 v20[38]; // [rsp+140h] [rbp-3C0h] BYREF
  char v21[8]; // [rsp+270h] [rbp-290h] BYREF
  unsigned __int64 v22; // [rsp+278h] [rbp-288h]
  char v23; // [rsp+28Ch] [rbp-274h]
  char *v24; // [rsp+2D0h] [rbp-230h]
  char v25; // [rsp+2E0h] [rbp-220h] BYREF
  char v26[8]; // [rsp+3A0h] [rbp-160h] BYREF
  unsigned __int64 v27; // [rsp+3A8h] [rbp-158h]
  char v28; // [rsp+3BCh] [rbp-144h]
  char *v29; // [rsp+400h] [rbp-100h]
  char v30; // [rsp+410h] [rbp-F0h] BYREF

  v11 = &v16;
  memset(v20, 0, sizeof(v20));
  v18 = 0x800000000LL;
  v20[1] = (unsigned __int64)&v20[4];
  v6 = *(_QWORD *)(a2 + 96);
  v20[12] = (unsigned __int64)&v20[14];
  v16 = v6;
  HIDWORD(v20[13]) = 8;
  v17 = v19;
  v7 = *(unsigned int *)(v6 + 32);
  BYTE4(v20[3]) = 1;
  v14 = 0;
  v15 = 1;
  v8 = *(_QWORD *)(v6 + 24);
  v19[2] = v6;
  v19[0] = v8 + 8 * v7;
  v19[1] = v8;
  LODWORD(v20[2]) = 8;
  v12 = 8;
  v13 = 1;
  v10 = 1;
  LODWORD(v18) = 1;
  sub_2DB5710((__int64)&v10, a2, v8, v19[0], (__int64)&v10, a6);
  sub_2DB59D0((__int64)v26, (__int64)v20);
  sub_2DB59D0((__int64)v21, (__int64)&v10);
  sub_2DB59D0(a1, (__int64)v21);
  sub_2DB59D0(a1 + 304, (__int64)v26);
  if ( v24 != &v25 )
    _libc_free((unsigned __int64)v24);
  if ( !v23 )
    _libc_free(v22);
  if ( v29 != &v30 )
    _libc_free((unsigned __int64)v29);
  if ( !v28 )
    _libc_free(v27);
  if ( v17 != v19 )
    _libc_free((unsigned __int64)v17);
  if ( !v15 )
    _libc_free((unsigned __int64)v11);
  if ( (unsigned __int64 *)v20[12] != &v20[14] )
    _libc_free(v20[12]);
  if ( !BYTE4(v20[3]) )
    _libc_free(v20[1]);
  return a1;
}
