// Function: sub_190E970
// Address: 0x190e970
//
void __fastcall sub_190E970(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // r12
  __m128i v8; // xmm0
  __int64 v9; // rax
  __m128i v10; // xmm1
  __int64 v11; // rax
  int v12; // eax
  char *v13; // r13
  char *v14; // r12
  char *v15; // rdi
  _QWORD *v16; // r13
  _QWORD *v17; // r12
  _QWORD *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  _QWORD v21[2]; // [rsp+0h] [rbp-4B0h] BYREF
  __int64 v22; // [rsp+10h] [rbp-4A0h] BYREF
  __int64 *v23; // [rsp+20h] [rbp-490h]
  __int64 v24; // [rsp+30h] [rbp-480h] BYREF
  _QWORD v25[2]; // [rsp+60h] [rbp-450h] BYREF
  __int64 v26; // [rsp+70h] [rbp-440h] BYREF
  __int64 *v27; // [rsp+80h] [rbp-430h]
  __int64 v28; // [rsp+90h] [rbp-420h] BYREF
  void *v29; // [rsp+C0h] [rbp-3F0h] BYREF
  int v30; // [rsp+C8h] [rbp-3E8h]
  char v31; // [rsp+CCh] [rbp-3E4h]
  __int64 v32; // [rsp+D0h] [rbp-3E0h]
  __m128i v33; // [rsp+D8h] [rbp-3D8h]
  __int64 v34; // [rsp+E8h] [rbp-3C8h]
  __int64 v35; // [rsp+F0h] [rbp-3C0h]
  __m128i v36; // [rsp+F8h] [rbp-3B8h]
  __int64 v37; // [rsp+108h] [rbp-3A8h]
  _BYTE *v39; // [rsp+118h] [rbp-398h] BYREF
  __int64 v40; // [rsp+120h] [rbp-390h]
  _BYTE v41[356]; // [rsp+128h] [rbp-388h] BYREF
  int v42; // [rsp+28Ch] [rbp-224h]
  __int64 v43; // [rsp+290h] [rbp-220h]
  _QWORD v44[11]; // [rsp+2A0h] [rbp-210h] BYREF
  char *v45; // [rsp+2F8h] [rbp-1B8h]
  unsigned int v46; // [rsp+300h] [rbp-1B0h]
  char v47; // [rsp+308h] [rbp-1A8h] BYREF

  v5 = sub_15E0530(*a1);
  if ( sub_1602790(v5)
    || (v19 = sub_15E0530(*a1),
        v20 = sub_16033E0(v19),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v20 + 48LL))(v20)) )
  {
    sub_15CA3B0((__int64)v44, (__int64)"gvn", (__int64)"LoadElim", 8, *a2);
    sub_15CAB20((__int64)v44, "load of type ", 0xDu);
    sub_15C9730((__int64)v25, "Type", 4, *(_QWORD *)*a2);
    v6 = sub_17C2270((__int64)v44, (__int64)v25);
    sub_15CAB20(v6, " eliminated", 0xBu);
    sub_15CA8D0(v6);
    sub_15CAB20(v6, " in favor of ", 0xDu);
    sub_15C9340((__int64)v21, "InfavorOfValue", 0xEu, *a3);
    v7 = sub_17C2270(v6, (__int64)v21);
    v8 = _mm_loadu_si128((const __m128i *)(v7 + 24));
    v30 = *(_DWORD *)(v7 + 8);
    v31 = *(_BYTE *)(v7 + 12);
    v9 = *(_QWORD *)(v7 + 16);
    v33 = v8;
    v32 = v9;
    v34 = *(_QWORD *)(v7 + 40);
    v10 = _mm_loadu_si128((const __m128i *)(v7 + 56));
    v29 = &unk_49ECF68;
    v11 = *(_QWORD *)(v7 + 48);
    v36 = v10;
    v35 = v11;
    if ( *(_BYTE *)(v7 + 80) )
      v37 = *(_QWORD *)(v7 + 72);
    v40 = 0x400000000LL;
    v12 = *(_DWORD *)(v7 + 96);
    v39 = v41;
    if ( v12 )
      sub_190E6E0((__int64)&v39, v7 + 88);
    v41[352] = *(_BYTE *)(v7 + 456);
    v42 = *(_DWORD *)(v7 + 460);
    v43 = *(_QWORD *)(v7 + 464);
    v29 = &unk_49ECF98;
    if ( v23 != &v24 )
      j_j___libc_free_0(v23, v24 + 1);
    if ( (__int64 *)v21[0] != &v22 )
      j_j___libc_free_0(v21[0], v22 + 1);
    if ( v27 != &v28 )
      j_j___libc_free_0(v27, v28 + 1);
    if ( (__int64 *)v25[0] != &v26 )
      j_j___libc_free_0(v25[0], v26 + 1);
    v13 = v45;
    v44[0] = &unk_49ECF68;
    v14 = &v45[88 * v46];
    if ( v45 != v14 )
    {
      do
      {
        v14 -= 88;
        v15 = (char *)*((_QWORD *)v14 + 4);
        if ( v15 != v14 + 48 )
          j_j___libc_free_0(v15, *((_QWORD *)v14 + 6) + 1LL);
        if ( *(char **)v14 != v14 + 16 )
          j_j___libc_free_0(*(_QWORD *)v14, *((_QWORD *)v14 + 2) + 1LL);
      }
      while ( v13 != v14 );
      v14 = v45;
    }
    if ( v14 != &v47 )
      _libc_free((unsigned __int64)v14);
    sub_143AA50(a1, (__int64)&v29);
    v16 = v39;
    v29 = &unk_49ECF68;
    v17 = &v39[88 * (unsigned int)v40];
    if ( v39 != (_BYTE *)v17 )
    {
      do
      {
        v17 -= 11;
        v18 = (_QWORD *)v17[4];
        if ( v18 != v17 + 6 )
          j_j___libc_free_0(v18, v17[6] + 1LL);
        if ( (_QWORD *)*v17 != v17 + 2 )
          j_j___libc_free_0(*v17, v17[2] + 1LL);
      }
      while ( v16 != v17 );
      v17 = v39;
    }
    if ( v17 != (_QWORD *)v41 )
      _libc_free((unsigned __int64)v17);
  }
}
