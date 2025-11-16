// Function: sub_38519E0
// Address: 0x38519e0
//
void __fastcall sub_38519E0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  __m128i v5; // xmm0
  __int64 v6; // rax
  __m128i v7; // xmm1
  __int64 v8; // rax
  int v9; // eax
  unsigned __int64 *v10; // r13
  unsigned __int64 *v11; // r12
  unsigned __int64 v12; // rdi
  unsigned __int64 *v13; // r13
  unsigned __int64 *v14; // r12
  unsigned __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18[2]; // [rsp+0h] [rbp-450h] BYREF
  __int64 v19; // [rsp+10h] [rbp-440h] BYREF
  __int64 *v20; // [rsp+20h] [rbp-430h]
  __int64 v21; // [rsp+30h] [rbp-420h] BYREF
  void *v22; // [rsp+60h] [rbp-3F0h] BYREF
  int v23; // [rsp+68h] [rbp-3E8h]
  char v24; // [rsp+6Ch] [rbp-3E4h]
  __int64 v25; // [rsp+70h] [rbp-3E0h]
  __m128i v26; // [rsp+78h] [rbp-3D8h]
  __int64 v27; // [rsp+88h] [rbp-3C8h]
  __int64 v28; // [rsp+90h] [rbp-3C0h]
  __m128i v29; // [rsp+98h] [rbp-3B8h]
  __int64 v30; // [rsp+A8h] [rbp-3A8h]
  unsigned __int64 *v32; // [rsp+B8h] [rbp-398h] BYREF
  __int64 v33; // [rsp+C0h] [rbp-390h]
  _BYTE v34[356]; // [rsp+C8h] [rbp-388h] BYREF
  int v35; // [rsp+22Ch] [rbp-224h]
  __int64 v36; // [rsp+230h] [rbp-220h]
  _QWORD v37[11]; // [rsp+240h] [rbp-210h] BYREF
  unsigned __int64 *v38; // [rsp+298h] [rbp-1B8h]
  unsigned int v39; // [rsp+2A0h] [rbp-1B0h]
  char v40; // [rsp+2A8h] [rbp-1A8h] BYREF

  v3 = sub_15E0530(*a1);
  if ( sub_1602790(v3)
    || (v16 = sub_15E0530(*a1),
        v17 = sub_16033E0(v16),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v17 + 48LL))(v17)) )
  {
    sub_15CA5C0(
      (__int64)v37,
      (__int64)"inline-cost",
      (__int64)"NeverInline",
      11,
      *(_QWORD *)(a2 + 56) & 0xFFFFFFFFFFFFFFF8LL);
    sub_15C9340((__int64)v18, "Callee", 6u, *(_QWORD *)(a2 + 32));
    v4 = sub_186AF50((__int64)v37, (__int64)v18);
    sub_15CAB20(v4, " has uninlinable pattern and cost is not fully computed", 0x37u);
    v5 = _mm_loadu_si128((const __m128i *)(v4 + 24));
    v23 = *(_DWORD *)(v4 + 8);
    v24 = *(_BYTE *)(v4 + 12);
    v6 = *(_QWORD *)(v4 + 16);
    v26 = v5;
    v25 = v6;
    v27 = *(_QWORD *)(v4 + 40);
    v7 = _mm_loadu_si128((const __m128i *)(v4 + 56));
    v22 = &unk_49ECF68;
    v8 = *(_QWORD *)(v4 + 48);
    v29 = v7;
    v28 = v8;
    if ( *(_BYTE *)(v4 + 80) )
      v30 = *(_QWORD *)(v4 + 72);
    v33 = 0x400000000LL;
    v9 = *(_DWORD *)(v4 + 96);
    v32 = (unsigned __int64 *)v34;
    if ( v9 )
      sub_3851750((__int64)&v32, v4 + 88);
    v34[352] = *(_BYTE *)(v4 + 456);
    v35 = *(_DWORD *)(v4 + 460);
    v36 = *(_QWORD *)(v4 + 464);
    v22 = &unk_49ECFC8;
    if ( v20 != &v21 )
      j_j___libc_free_0((unsigned __int64)v20);
    if ( (__int64 *)v18[0] != &v19 )
      j_j___libc_free_0(v18[0]);
    v10 = v38;
    v37[0] = &unk_49ECF68;
    v11 = &v38[11 * v39];
    if ( v38 != v11 )
    {
      do
      {
        v11 -= 11;
        v12 = v11[4];
        if ( (unsigned __int64 *)v12 != v11 + 6 )
          j_j___libc_free_0(v12);
        if ( (unsigned __int64 *)*v11 != v11 + 2 )
          j_j___libc_free_0(*v11);
      }
      while ( v10 != v11 );
      v11 = v38;
    }
    if ( v11 != (unsigned __int64 *)&v40 )
      _libc_free((unsigned __int64)v11);
    sub_143AA50(a1, (__int64)&v22);
    v13 = v32;
    v22 = &unk_49ECF68;
    v14 = &v32[11 * (unsigned int)v33];
    if ( v32 != v14 )
    {
      do
      {
        v14 -= 11;
        v15 = v14[4];
        if ( (unsigned __int64 *)v15 != v14 + 6 )
          j_j___libc_free_0(v15);
        if ( (unsigned __int64 *)*v14 != v14 + 2 )
          j_j___libc_free_0(*v14);
      }
      while ( v13 != v14 );
      v14 = v32;
    }
    if ( v14 != (unsigned __int64 *)v34 )
      _libc_free((unsigned __int64)v14);
  }
}
