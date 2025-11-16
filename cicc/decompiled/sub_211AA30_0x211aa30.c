// Function: sub_211AA30
// Address: 0x211aa30
//
__int64 __fastcall sub_211AA30(__int64 *a1, _QWORD *a2, unsigned int a3, __m128i a4, double a5, __m128i a6)
{
  __int64 v6; // rbx
  unsigned __int8 *v7; // rax
  __int64 v8; // r14
  _BYTE *v9; // r14
  char *v10; // rax
  unsigned __int8 v11; // dl
  __int64 v12; // r15
  __int64 v13; // rsi
  const void **v14; // r8
  unsigned int v15; // r12d
  __int64 v16; // rax
  void *v17; // rbx
  __int64 *v18; // rsi
  __int64 *v20; // rsi
  __int64 *v21; // r9
  void *v22; // rcx
  __int64 *v23; // rsi
  __int64 *v24; // r9
  __int64 v25; // r12
  _QWORD *v26; // rsi
  const void **v27; // r8
  unsigned int v28; // r15d
  const void **v29; // [rsp+0h] [rbp-80h]
  void *v30; // [rsp+8h] [rbp-78h]
  const void **v31; // [rsp+8h] [rbp-78h]
  __int64 *v32; // [rsp+8h] [rbp-78h]
  __int64 *v33; // [rsp+8h] [rbp-78h]
  const void **v34; // [rsp+8h] [rbp-78h]
  __int64 v35; // [rsp+10h] [rbp-70h] BYREF
  _QWORD *v36; // [rsp+18h] [rbp-68h]
  __int64 v37; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v38; // [rsp+28h] [rbp-58h]
  _QWORD *v39; // [rsp+30h] [rbp-50h] BYREF
  __int64 v40; // [rsp+38h] [rbp-48h]
  const void **v41; // [rsp+40h] [rbp-40h]

  v6 = (__int64)a2;
  v7 = (unsigned __int8 *)(a2[5] + 16LL * a3);
  v8 = *v7;
  sub_1F40D10((__int64)&v39, *a1, *(_QWORD *)(a1[1] + 48), (unsigned __int8)v8, *((_QWORD *)v7 + 1));
  if ( (_BYTE)v8 != (_BYTE)v40 || !(_BYTE)v8 || !*(_QWORD *)(*a1 + 8 * v8 + 120) )
  {
    v9 = (_BYTE *)sub_1E0A0C0(*(_QWORD *)(a1[1] + 32));
    v30 = sub_16982C0();
    v10 = (char *)a2[5];
    v11 = *v10;
    if ( *v9 && v11 == 13 )
    {
      v20 = (__int64 *)(a2[11] + 32LL);
      if ( v30 == (void *)*v20 )
        sub_169D930((__int64)&v39, (__int64)v20);
      else
        sub_169D7E0((__int64)&v39, v20);
      v21 = a1;
      if ( (unsigned int)v40 <= 0x40 )
      {
        v35 = v40;
      }
      else
      {
        v35 = v39[1];
        j_j___libc_free_0_0(v39);
        v21 = a1;
      }
      v22 = v30;
      v32 = v21;
      v23 = (__int64 *)(*(_QWORD *)(v6 + 88) + 32LL);
      if ( v22 == (void *)*v23 )
        sub_169D930((__int64)&v39, (__int64)v23);
      else
        sub_169D7E0((__int64)&v39, v23);
      v24 = v32;
      if ( (unsigned int)v40 <= 0x40 )
      {
        v36 = v39;
      }
      else
      {
        v36 = (_QWORD *)*v39;
        j_j___libc_free_0_0(v39);
        v24 = v32;
      }
      v33 = v24;
      sub_16A50F0((__int64)&v37, 128, &v35, 2u);
      v25 = v33[1];
      sub_1F40D10(
        (__int64)&v39,
        *v33,
        *(_QWORD *)(v25 + 48),
        **(unsigned __int8 **)(v6 + 40),
        *(_QWORD *)(*(_QWORD *)(v6 + 40) + 8LL));
      v26 = *(_QWORD **)(v6 + 72);
      v27 = v41;
      v28 = (unsigned __int8)v40;
      v39 = v26;
      if ( v26 )
      {
        v34 = v41;
        sub_1623A60((__int64)&v39, (__int64)v26, 2);
        v27 = v34;
      }
      LODWORD(v40) = *(_DWORD *)(v6 + 64);
      v6 = sub_1D38970(v25, (__int64)&v37, (__int64)&v39, v28, v27, 0, a4, a5, a6, 0);
      if ( v39 )
        sub_161E7C0((__int64)&v39, (__int64)v39);
      if ( v38 > 0x40 && v37 )
        j_j___libc_free_0_0(v37);
    }
    else
    {
      v12 = a1[1];
      sub_1F40D10((__int64)&v39, *a1, *(_QWORD *)(v12 + 48), v11, *((_QWORD *)v10 + 1));
      v13 = a2[9];
      v14 = v41;
      v15 = (unsigned __int8)v40;
      v37 = v13;
      if ( v13 )
      {
        v29 = v41;
        sub_1623A60((__int64)&v37, v13, 2);
        v14 = v29;
      }
      v38 = *(_DWORD *)(v6 + 64);
      v16 = *(_QWORD *)(v6 + 88);
      v17 = v30;
      v31 = v14;
      v18 = (__int64 *)(v16 + 32);
      if ( v17 == *(void **)(v16 + 32) )
        sub_169D930((__int64)&v39, (__int64)v18);
      else
        sub_169D7E0((__int64)&v39, v18);
      v6 = sub_1D38970(v12, (__int64)&v39, (__int64)&v37, v15, v31, 0, a4, a5, a6, 0);
      if ( (unsigned int)v40 > 0x40 && v39 )
        j_j___libc_free_0_0(v39);
      if ( v37 )
        sub_161E7C0((__int64)&v37, v37);
    }
  }
  return v6;
}
