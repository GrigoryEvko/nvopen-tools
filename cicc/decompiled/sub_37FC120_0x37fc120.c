// Function: sub_37FC120
// Address: 0x37fc120
//
unsigned __int8 *__fastcall sub_37FC120(__int64 *a1, __int64 a2, __m128i a3)
{
  _BYTE *v4; // r14
  void *v5; // r13
  __int16 *v6; // rax
  unsigned __int16 v7; // si
  __int64 v8; // r15
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int, __int64); // r10
  unsigned int v10; // r12d
  __int64 v11; // r8
  _QWORD *v12; // rsi
  __int64 *v13; // rsi
  unsigned __int8 *v14; // r12
  __int64 *v16; // rsi
  __int64 *v17; // rsi
  __int64 v18; // r15
  __int64 (__fastcall *v19)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v20; // rax
  unsigned __int16 v21; // si
  unsigned int v22; // r12d
  __int64 v23; // r8
  _QWORD *v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // [rsp+8h] [rbp-78h]
  __int64 v28; // [rsp+8h] [rbp-78h]
  __int64 v29; // [rsp+8h] [rbp-78h]
  __int64 v30; // [rsp+10h] [rbp-70h] BYREF
  _QWORD *v31; // [rsp+18h] [rbp-68h]
  unsigned __int64 v32; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v33; // [rsp+28h] [rbp-58h]
  _QWORD *v34; // [rsp+30h] [rbp-50h] BYREF
  __int64 v35; // [rsp+38h] [rbp-48h]
  __int64 v36; // [rsp+40h] [rbp-40h]

  v4 = (_BYTE *)sub_2E79000(*(__int64 **)(a1[1] + 40));
  v5 = sub_C33340();
  v6 = *(__int16 **)(a2 + 48);
  v7 = *v6;
  if ( *v4 && v7 == 16 )
  {
    v16 = (__int64 *)(*(_QWORD *)(a2 + 96) + 24LL);
    if ( (void *)*v16 == v5 )
      sub_C3E660((__int64)&v34, (__int64)v16);
    else
      sub_C3A850((__int64)&v34, v16);
    if ( (unsigned int)v35 <= 0x40 )
    {
      v30 = v35;
    }
    else
    {
      v30 = v34[1];
      j_j___libc_free_0_0((unsigned __int64)v34);
    }
    v17 = (__int64 *)(*(_QWORD *)(a2 + 96) + 24LL);
    if ( (void *)*v17 == v5 )
      sub_C3E660((__int64)&v34, (__int64)v17);
    else
      sub_C3A850((__int64)&v34, v17);
    if ( (unsigned int)v35 <= 0x40 )
    {
      v31 = v34;
    }
    else
    {
      v31 = (_QWORD *)*v34;
      j_j___libc_free_0_0((unsigned __int64)v34);
    }
    sub_C438C0((__int64)&v32, 128, &v30, 2u);
    v18 = a1[1];
    v19 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
    v20 = *(__int16 **)(a2 + 48);
    v21 = *v20;
    if ( v19 == sub_2D56A50 )
    {
      HIWORD(v22) = 0;
      sub_2FE6CC0((__int64)&v34, *a1, *(_QWORD *)(v18 + 64), v21, *((_QWORD *)v20 + 1));
      LOWORD(v22) = v35;
      v23 = v36;
    }
    else
    {
      v22 = v19(*a1, *(_QWORD *)(v18 + 64), v21, *((_QWORD *)v20 + 1));
      v23 = v26;
    }
    v24 = *(_QWORD **)(a2 + 80);
    v34 = v24;
    if ( v24 )
    {
      v29 = v23;
      sub_B96E90((__int64)&v34, (__int64)v24, 1);
      v23 = v29;
    }
    LODWORD(v35) = *(_DWORD *)(a2 + 72);
    v14 = sub_34007B0(v18, (__int64)&v32, (__int64)&v34, v22, v23, 0, a3, 0);
    if ( v34 )
      sub_B91220((__int64)&v34, (__int64)v34);
    if ( v33 > 0x40 && v32 )
      j_j___libc_free_0_0(v32);
  }
  else
  {
    v8 = a1[1];
    v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
    if ( v9 == sub_2D56A50 )
    {
      HIWORD(v10) = 0;
      sub_2FE6CC0((__int64)&v34, *a1, *(_QWORD *)(v8 + 64), v7, *((_QWORD *)v6 + 1));
      LOWORD(v10) = v35;
      v11 = v36;
    }
    else
    {
      v10 = v9(*a1, *(_QWORD *)(v8 + 64), v7, *((_QWORD *)v6 + 1));
      v11 = v25;
    }
    v12 = *(_QWORD **)(a2 + 80);
    v34 = v12;
    if ( v12 )
    {
      v27 = v11;
      sub_B96E90((__int64)&v34, (__int64)v12, 1);
      v11 = v27;
    }
    v28 = v11;
    LODWORD(v35) = *(_DWORD *)(a2 + 72);
    v13 = (__int64 *)(*(_QWORD *)(a2 + 96) + 24LL);
    if ( (void *)*v13 == v5 )
      sub_C3E660((__int64)&v32, (__int64)v13);
    else
      sub_C3A850((__int64)&v32, v13);
    v14 = sub_34007B0(v8, (__int64)&v32, (__int64)&v34, v10, v28, 0, a3, 0);
    if ( v33 > 0x40 && v32 )
      j_j___libc_free_0_0(v32);
    if ( v34 )
      sub_B91220((__int64)&v34, (__int64)v34);
  }
  return v14;
}
