// Function: sub_20600C0
// Address: 0x20600c0
//
void __fastcall sub_20600C0(__int64 *a1, __int64 a2, __int64 a3, double a4, double a5, __m128i a6)
{
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // r14
  _QWORD *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // r8
  int v17; // r8d
  _QWORD *v18; // r12
  __int64 v19; // rdx
  __int64 v20; // r13
  __int64 v21; // rax
  _QWORD *v22; // rax
  unsigned __int64 v23; // rdx
  int v24; // ecx
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 v27; // rsi
  __int64 *v28; // r12
  int v29; // edx
  __int64 *v30; // rax
  unsigned __int64 v31; // [rsp+10h] [rbp-C0h]
  __int64 v32; // [rsp+28h] [rbp-A8h]
  int v33; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v34; // [rsp+48h] [rbp-88h] BYREF
  __int64 v35; // [rsp+50h] [rbp-80h] BYREF
  int v36; // [rsp+58h] [rbp-78h]
  _BYTE *v37; // [rsp+60h] [rbp-70h] BYREF
  __int64 v38; // [rsp+68h] [rbp-68h]
  _BYTE v39[16]; // [rsp+70h] [rbp-60h] BYREF
  _BYTE *v40; // [rsp+80h] [rbp-50h] BYREF
  __int64 v41; // [rsp+88h] [rbp-48h]
  _BYTE v42[64]; // [rsp+90h] [rbp-40h] BYREF

  v31 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  sub_1602B00(*(_QWORD *)(a1[69] + 48), a2 & 0xFFFFFFFFFFFFFFF8LL, a3);
  v7 = *(_QWORD *)(a2 & 0xFFFFFFFFFFFFFFF8LL);
  v8 = a1[69];
  v9 = *(_QWORD *)(v8 + 32);
  v10 = *(_QWORD *)(v8 + 16);
  v37 = v39;
  v38 = 0x100000000LL;
  v11 = sub_1E0A0C0(v9);
  sub_20C7CE0(v10, v11, v7, &v37, 0, 0);
  if ( (_DWORD)v38 )
  {
    v41 = 0x100000000LL;
    v40 = v42;
    v13 = 0;
    v32 = 16LL * (unsigned int)v38;
    do
    {
      v14 = (_QWORD *)a1[69];
      v15 = *(_QWORD *)&v37[v13];
      v16 = *(_QWORD *)&v37[v13 + 8];
      v36 = 0;
      v35 = 0;
      v18 = sub_1D2B300(v14, 0x30u, (__int64)&v35, v15, v16, v12);
      v20 = v19;
      if ( v35 )
        sub_161E7C0((__int64)&v35, v35);
      v21 = (unsigned int)v41;
      if ( (unsigned int)v41 >= HIDWORD(v41) )
      {
        sub_16CD150((__int64)&v40, v42, 0, 16, v17, v12);
        v21 = (unsigned int)v41;
      }
      v22 = &v40[16 * v21];
      v13 += 16;
      *v22 = v18;
      v22[1] = v20;
      v23 = (unsigned int)(v41 + 1);
      LODWORD(v41) = v41 + 1;
    }
    while ( v13 != v32 );
    v24 = *((_DWORD *)a1 + 134);
    v25 = *a1;
    v35 = 0;
    v26 = a1[69];
    v36 = v24;
    if ( v25 )
    {
      if ( &v35 != (__int64 *)(v25 + 48) )
      {
        v27 = *(_QWORD *)(v25 + 48);
        v35 = v27;
        if ( v27 )
        {
          sub_1623A60((__int64)&v35, v27, 2);
          v23 = (unsigned int)v41;
        }
      }
    }
    v28 = sub_1D37190(v26, (__int64)v40, v23, (__int64)&v35, v17, a4, a5, a6);
    v33 = v29;
    v34 = v31;
    v30 = sub_205F5C0((__int64)(a1 + 1), (__int64 *)&v34);
    v30[1] = (__int64)v28;
    *((_DWORD *)v30 + 4) = v33;
    if ( v35 )
      sub_161E7C0((__int64)&v35, v35);
    if ( v40 != v42 )
      _libc_free((unsigned __int64)v40);
  }
  if ( v37 != v39 )
    _libc_free((unsigned __int64)v37);
}
