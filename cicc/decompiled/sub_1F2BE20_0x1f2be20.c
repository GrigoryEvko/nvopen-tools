// Function: sub_1F2BE20
// Address: 0x1f2be20
//
void __fastcall sub_1F2BE20(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r12
  __m128i v7; // xmm0
  __int64 v8; // rax
  __m128i v9; // xmm1
  __int64 v10; // rax
  int v11; // eax
  char *v12; // r13
  char *v13; // r12
  char *v14; // rdi
  _QWORD *v15; // r13
  _QWORD *v16; // r12
  _QWORD *v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rax
  _QWORD v20[2]; // [rsp+0h] [rbp-450h] BYREF
  __int64 v21; // [rsp+10h] [rbp-440h] BYREF
  __int64 *v22; // [rsp+20h] [rbp-430h]
  __int64 v23; // [rsp+30h] [rbp-420h] BYREF
  void *v24; // [rsp+60h] [rbp-3F0h] BYREF
  int v25; // [rsp+68h] [rbp-3E8h]
  char v26; // [rsp+6Ch] [rbp-3E4h]
  __int64 v27; // [rsp+70h] [rbp-3E0h]
  __m128i v28; // [rsp+78h] [rbp-3D8h]
  __int64 v29; // [rsp+88h] [rbp-3C8h]
  __int64 v30; // [rsp+90h] [rbp-3C0h]
  __m128i v31; // [rsp+98h] [rbp-3B8h]
  __int64 v32; // [rsp+A8h] [rbp-3A8h]
  _BYTE *v34; // [rsp+B8h] [rbp-398h] BYREF
  __int64 v35; // [rsp+C0h] [rbp-390h]
  _BYTE v36[356]; // [rsp+C8h] [rbp-388h] BYREF
  int v37; // [rsp+22Ch] [rbp-224h]
  __int64 v38; // [rsp+230h] [rbp-220h]
  _QWORD v39[11]; // [rsp+240h] [rbp-210h] BYREF
  char *v40; // [rsp+298h] [rbp-1B8h]
  unsigned int v41; // [rsp+2A0h] [rbp-1B0h]
  char v42; // [rsp+2A8h] [rbp-1A8h] BYREF

  v5 = sub_15E0530(*a1);
  if ( sub_1602790(v5)
    || (v18 = sub_15E0530(*a1),
        v19 = sub_16033E0(v18),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v19 + 48LL))(v19)) )
  {
    sub_15CA3B0((__int64)v39, (__int64)"stack-protector", (__int64)"StackProtectorAllocaOrArray", 27, a2);
    sub_15CAB20((__int64)v39, "Stack protection applied to function ", 0x25u);
    sub_15C9340((__int64)v20, "Function", 8u, *(_QWORD *)(a3 + 232));
    v6 = sub_17C2270((__int64)v39, (__int64)v20);
    sub_15CAB20(v6, " due to a call to alloca or use of a variable length array", 0x3Au);
    v7 = _mm_loadu_si128((const __m128i *)(v6 + 24));
    v25 = *(_DWORD *)(v6 + 8);
    v26 = *(_BYTE *)(v6 + 12);
    v8 = *(_QWORD *)(v6 + 16);
    v28 = v7;
    v27 = v8;
    v29 = *(_QWORD *)(v6 + 40);
    v9 = _mm_loadu_si128((const __m128i *)(v6 + 56));
    v24 = &unk_49ECF68;
    v10 = *(_QWORD *)(v6 + 48);
    v31 = v9;
    v30 = v10;
    if ( *(_BYTE *)(v6 + 80) )
      v32 = *(_QWORD *)(v6 + 72);
    v35 = 0x400000000LL;
    v11 = *(_DWORD *)(v6 + 96);
    v34 = v36;
    if ( v11 )
      sub_1F2BB90((__int64)&v34, v6 + 88);
    v36[352] = *(_BYTE *)(v6 + 456);
    v37 = *(_DWORD *)(v6 + 460);
    v38 = *(_QWORD *)(v6 + 464);
    v24 = &unk_49ECF98;
    if ( v22 != &v23 )
      j_j___libc_free_0(v22, v23 + 1);
    if ( (__int64 *)v20[0] != &v21 )
      j_j___libc_free_0(v20[0], v21 + 1);
    v12 = v40;
    v39[0] = &unk_49ECF68;
    v13 = &v40[88 * v41];
    if ( v40 != v13 )
    {
      do
      {
        v13 -= 88;
        v14 = (char *)*((_QWORD *)v13 + 4);
        if ( v14 != v13 + 48 )
          j_j___libc_free_0(v14, *((_QWORD *)v13 + 6) + 1LL);
        if ( *(char **)v13 != v13 + 16 )
          j_j___libc_free_0(*(_QWORD *)v13, *((_QWORD *)v13 + 2) + 1LL);
      }
      while ( v12 != v13 );
      v13 = v40;
    }
    if ( v13 != &v42 )
      _libc_free((unsigned __int64)v13);
    sub_143AA50(a1, (__int64)&v24);
    v15 = v34;
    v24 = &unk_49ECF68;
    v16 = &v34[88 * (unsigned int)v35];
    if ( v34 != (_BYTE *)v16 )
    {
      do
      {
        v16 -= 11;
        v17 = (_QWORD *)v16[4];
        if ( v17 != v16 + 6 )
          j_j___libc_free_0(v17, v16[6] + 1LL);
        if ( (_QWORD *)*v16 != v16 + 2 )
          j_j___libc_free_0(*v16, v16[2] + 1LL);
      }
      while ( v15 != v16 );
      v16 = v34;
    }
    if ( v16 != (_QWORD *)v36 )
      _libc_free((unsigned __int64)v16);
  }
}
