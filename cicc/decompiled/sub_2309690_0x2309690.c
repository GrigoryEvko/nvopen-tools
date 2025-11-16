// Function: sub_2309690
// Address: 0x2309690
//
__int64 *__fastcall sub_2309690(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r14
  unsigned __int64 v11; // rdi
  bool v12; // zf
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r12
  unsigned __int64 v16; // rdi
  _QWORD v18[3]; // [rsp+10h] [rbp-430h] BYREF
  char v19; // [rsp+28h] [rbp-418h]
  char v20[8]; // [rsp+30h] [rbp-410h] BYREF
  unsigned __int64 v21; // [rsp+38h] [rbp-408h]
  char v22; // [rsp+4Ch] [rbp-3F4h]
  char v23[256]; // [rsp+50h] [rbp-3F0h] BYREF
  __int64 v24; // [rsp+150h] [rbp-2F0h]
  __int64 v25; // [rsp+158h] [rbp-2E8h]
  __int64 v26; // [rsp+160h] [rbp-2E0h]
  unsigned int v27; // [rsp+168h] [rbp-2D8h]
  char v28[8]; // [rsp+170h] [rbp-2D0h] BYREF
  unsigned __int64 v29; // [rsp+178h] [rbp-2C8h]
  char v30; // [rsp+18Ch] [rbp-2B4h]
  char v31[128]; // [rsp+190h] [rbp-2B0h] BYREF
  __int64 v32; // [rsp+210h] [rbp-230h]
  __int64 v33; // [rsp+218h] [rbp-228h]
  __int64 v34; // [rsp+220h] [rbp-220h]
  char v35; // [rsp+228h] [rbp-218h]
  char v36[8]; // [rsp+230h] [rbp-210h] BYREF
  unsigned __int64 v37; // [rsp+238h] [rbp-208h]
  char v38; // [rsp+24Ch] [rbp-1F4h]
  _BYTE v39[256]; // [rsp+250h] [rbp-1F0h] BYREF
  __int64 v40; // [rsp+350h] [rbp-F0h]
  __int64 v41; // [rsp+358h] [rbp-E8h]
  __int64 v42; // [rsp+360h] [rbp-E0h]
  unsigned int v43; // [rsp+368h] [rbp-D8h]
  char v44[8]; // [rsp+370h] [rbp-D0h] BYREF
  unsigned __int64 v45; // [rsp+378h] [rbp-C8h]
  char v46; // [rsp+38Ch] [rbp-B4h]
  _BYTE v47[176]; // [rsp+390h] [rbp-B0h] BYREF

  sub_D174E0((__int64)v18, a2 + 8, a3, a4);
  v32 = v18[0];
  v33 = v18[1];
  v34 = v18[2];
  v35 = v19;
  sub_C8CF70((__int64)v36, v39, 32, (__int64)v23, (__int64)v20);
  ++v24;
  v40 = 1;
  v41 = v25;
  v25 = 0;
  v42 = v26;
  v26 = 0;
  v43 = v27;
  v27 = 0;
  sub_C8CF70((__int64)v44, v47, 16, (__int64)v31, (__int64)v28);
  v5 = sub_22077B0(0x208u);
  v6 = v5;
  if ( v5 )
  {
    *(_QWORD *)v5 = &unk_4A0B088;
    *(_QWORD *)(v5 + 8) = v32;
    *(_QWORD *)(v5 + 16) = v33;
    *(_QWORD *)(v5 + 24) = v34;
    *(_BYTE *)(v5 + 32) = v35;
    sub_C8CF70(v5 + 40, (void *)(v5 + 72), 32, (__int64)v39, (__int64)v36);
    v7 = v41;
    *(_QWORD *)(v6 + 328) = 1;
    ++v40;
    *(_QWORD *)(v6 + 336) = v7;
    v41 = 0;
    *(_QWORD *)(v6 + 344) = v42;
    v42 = 0;
    *(_DWORD *)(v6 + 352) = v43;
    v43 = 0;
    sub_C8CF70(v6 + 360, (void *)(v6 + 392), 16, (__int64)v47, (__int64)v44);
  }
  if ( !v46 )
    _libc_free(v45);
  v8 = v43;
  if ( v43 )
  {
    v9 = v41;
    v10 = v41 + 24LL * v43;
    do
    {
      if ( *(_QWORD *)v9 != -4096 && *(_QWORD *)v9 != -8192 && *(_DWORD *)(v9 + 16) > 0x40u )
      {
        v11 = *(_QWORD *)(v9 + 8);
        if ( v11 )
          j_j___libc_free_0_0(v11);
      }
      v9 += 24;
    }
    while ( v10 != v9 );
    v8 = v43;
  }
  sub_C7D6A0(v41, 24 * v8, 8);
  if ( !v38 )
    _libc_free(v37);
  v12 = v30 == 0;
  *a1 = v6;
  if ( v12 )
    _libc_free(v29);
  v13 = v27;
  if ( v27 )
  {
    v14 = v25;
    v15 = v25 + 24LL * v27;
    do
    {
      if ( *(_QWORD *)v14 != -8192 && *(_QWORD *)v14 != -4096 && *(_DWORD *)(v14 + 16) > 0x40u )
      {
        v16 = *(_QWORD *)(v14 + 8);
        if ( v16 )
          j_j___libc_free_0_0(v16);
      }
      v14 += 24;
    }
    while ( v15 != v14 );
    v13 = v27;
  }
  sub_C7D6A0(v25, 24 * v13, 8);
  if ( !v22 )
    _libc_free(v21);
  return a1;
}
