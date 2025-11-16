// Function: sub_94C5F0
// Address: 0x94c5f0
//
__int64 __fastcall sub_94C5F0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rax
  __m128i *v9; // rax
  __m128i *v10; // r12
  __int64 v11; // rdi
  __int64 (__fastcall *v12)(__int64, __int64, __int64); // rax
  unsigned int *v13; // r15
  unsigned int *v14; // r14
  __int64 v15; // rdx
  __int64 v16; // rsi
  __m128i *v17; // rax
  __m128i *v18; // r15
  __int64 v19; // rdi
  __int64 (__fastcall *v20)(__int64, __int64, __int64); // rax
  __int64 v21; // rax
  __int64 v22; // r14
  unsigned int *v23; // r14
  unsigned int *v24; // rbx
  __int64 v25; // rdx
  __int64 v26; // rsi
  __m128i *v27; // rax
  __int64 *v28; // rdi
  __m128i *v29; // r14
  __int64 v30; // rax
  unsigned __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v34; // rax
  __int64 v35; // r14
  unsigned int *v36; // r14
  unsigned int *v37; // rbx
  __int64 v38; // rdx
  __int64 v39; // rsi
  unsigned int *v40; // rax
  unsigned int *v41; // r14
  unsigned int *v42; // r15
  __int64 v43; // rdx
  __int64 v44; // rsi
  __int64 v45; // [rsp+0h] [rbp-F0h]
  __int64 v47; // [rsp+10h] [rbp-E0h]
  __int64 v48; // [rsp+18h] [rbp-D8h]
  __int64 v49; // [rsp+18h] [rbp-D8h]
  __int64 v50; // [rsp+30h] [rbp-C0h]
  __int64 v52; // [rsp+40h] [rbp-B0h]
  __int64 v53; // [rsp+48h] [rbp-A8h]
  _QWORD v54[2]; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD v55[4]; // [rsp+60h] [rbp-90h] BYREF
  __m128i *v56; // [rsp+80h] [rbp-70h]
  _BYTE v57[32]; // [rsp+90h] [rbp-60h] BYREF
  __int16 v58; // [rsp+B0h] [rbp-40h]

  v6 = a2;
  v7 = sub_BCB2B0(*(_QWORD *)(a2 + 40));
  v50 = sub_BCE760(v7, 3);
  v8 = sub_BCB2B0(*(_QWORD *)(a2 + 40));
  v53 = sub_BCE760(v8, 1);
  v48 = sub_BCB2D0(*(_QWORD *)(a2 + 40));
  v47 = sub_AD64C0(v48, 0, 0);
  LOWORD(v56) = 257;
  v52 = a2 + 48;
  v9 = sub_92F410(a2, a3);
  v10 = v9;
  if ( v50 != v9->m128i_i64[1] )
  {
    if ( v9->m128i_i8[0] > 0x15u )
    {
      v58 = 257;
      v10 = (__m128i *)sub_B52210(v9, v50, v57, 0, 0);
      (*(void (__fastcall **)(_QWORD, __m128i *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
        *(_QWORD *)(a2 + 136),
        v10,
        v55,
        *(_QWORD *)(a2 + 104),
        *(_QWORD *)(a2 + 112));
      v40 = *(unsigned int **)(a2 + 48);
      v41 = &v40[4 * *(unsigned int *)(a2 + 56)];
      if ( v40 != v41 )
      {
        v42 = *(unsigned int **)(a2 + 48);
        do
        {
          v43 = *((_QWORD *)v42 + 1);
          v44 = *v42;
          v42 += 4;
          sub_B99FD0(v10, v44, v43);
        }
        while ( v41 != v42 );
      }
    }
    else
    {
      v11 = *(_QWORD *)(a2 + 128);
      v12 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v11 + 136LL);
      if ( v12 == sub_928970 )
        v10 = (__m128i *)sub_ADAFB0(v10, v50);
      else
        v10 = (__m128i *)v12(v11, (__int64)v10, v50);
      if ( v10->m128i_i8[0] > 0x1Cu )
      {
        (*(void (__fastcall **)(_QWORD, __m128i *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
          *(_QWORD *)(a2 + 136),
          v10,
          v55,
          *(_QWORD *)(a2 + 104),
          *(_QWORD *)(a2 + 112));
        v13 = *(unsigned int **)(a2 + 48);
        v14 = &v13[4 * *(unsigned int *)(a2 + 56)];
        while ( v14 != v13 )
        {
          v15 = *((_QWORD *)v13 + 1);
          v16 = *v13;
          v13 += 4;
          sub_B99FD0(v10, v16, v15);
        }
      }
    }
  }
  LOWORD(v56) = 257;
  v17 = sub_92F410(v6, *(_QWORD *)(a3 + 16));
  v18 = v17;
  if ( v53 != v17->m128i_i64[1] )
  {
    if ( v17->m128i_i8[0] <= 0x15u )
    {
      v19 = *(_QWORD *)(v6 + 128);
      v20 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v19 + 136LL);
      if ( v20 == sub_928970 )
        v18 = (__m128i *)sub_ADAFB0(v18, v53);
      else
        v18 = (__m128i *)v20(v19, (__int64)v18, v53);
      if ( v18->m128i_i8[0] <= 0x1Cu )
        goto LABEL_17;
      (*(void (__fastcall **)(_QWORD, __m128i *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v6 + 136) + 16LL))(
        *(_QWORD *)(v6 + 136),
        v18,
        v55,
        *(_QWORD *)(v52 + 56),
        *(_QWORD *)(v52 + 64));
      v21 = *(_QWORD *)(v6 + 48);
      v22 = 16LL * *(unsigned int *)(v6 + 56);
      if ( v21 == v21 + v22 )
        goto LABEL_17;
      v45 = v6;
      v23 = (unsigned int *)(v21 + v22);
      v24 = *(unsigned int **)(v6 + 48);
      do
      {
        v25 = *((_QWORD *)v24 + 1);
        v26 = *v24;
        v24 += 4;
        sub_B99FD0(v18, v26, v25);
      }
      while ( v23 != v24 );
      goto LABEL_16;
    }
    v58 = 257;
    v18 = (__m128i *)sub_B52210(v17, v53, v57, 0, 0);
    (*(void (__fastcall **)(_QWORD, __m128i *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v6 + 136) + 16LL))(
      *(_QWORD *)(v6 + 136),
      v18,
      v55,
      *(_QWORD *)(v52 + 56),
      *(_QWORD *)(v52 + 64));
    v34 = *(_QWORD *)(v6 + 48);
    v35 = 16LL * *(unsigned int *)(v6 + 56);
    if ( v34 != v34 + v35 )
    {
      v45 = v6;
      v36 = (unsigned int *)(v34 + v35);
      v37 = *(unsigned int **)(v6 + 48);
      do
      {
        v38 = *((_QWORD *)v37 + 1);
        v39 = *v37;
        v37 += 4;
        sub_B99FD0(v18, v39, v38);
      }
      while ( v36 != v37 );
LABEL_16:
      v6 = v45;
    }
  }
LABEL_17:
  v49 = sub_AD64C0(v48, a4, 0);
  v27 = sub_92F410(v6, *(_QWORD *)(*(_QWORD *)(a3 + 16) + 16LL));
  v28 = *(__int64 **)(v6 + 32);
  v29 = v27;
  v54[0] = v50;
  v54[1] = v53;
  v30 = sub_90A810(v28, 9056, (__int64)v54, 2u);
  v55[2] = v18;
  v31 = 0;
  v55[1] = v10;
  v55[3] = v49;
  v55[0] = v47;
  v56 = v29;
  v58 = 257;
  if ( v30 )
    v31 = *(_QWORD *)(v30 + 24);
  v32 = sub_921880((unsigned int **)v52, v31, v30, (int)v55, 5, (__int64)v57, 0);
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = v32;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
