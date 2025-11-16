// Function: sub_94F430
// Address: 0x94f430
//
__int64 __fastcall sub_94F430(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4, int a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // r13
  unsigned int v8; // ebx
  __int64 v9; // rdi
  __int64 v10; // r14
  __int64 v11; // r15
  __m128i *v12; // r13
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  unsigned int **v17; // r14
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 (__fastcall *v20)(__int64, __int64, __int64); // rax
  unsigned int *v21; // r15
  unsigned int *v22; // rbx
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rdx
  _BYTE *v26; // rbx
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int64 *v29; // rdi
  __int64 v30; // rax
  unsigned __int64 v31; // rsi
  __int64 v32; // rsi
  _BYTE *v33; // rdi
  __int64 v35; // rax
  __int64 v36; // rbx
  unsigned int *v37; // rbx
  unsigned int *v38; // r12
  __int64 v39; // rdx
  __int64 v40; // rsi
  __m128i *v43; // [rsp+20h] [rbp-F0h]
  __m128i *v44; // [rsp+28h] [rbp-E8h]
  __int64 v46; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v47; // [rsp+48h] [rbp-C8h] BYREF
  _BYTE *v48; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v49; // [rsp+58h] [rbp-B8h]
  _BYTE v50[32]; // [rsp+60h] [rbp-B0h] BYREF
  _QWORD v51[4]; // [rsp+80h] [rbp-90h] BYREF
  __int16 v52; // [rsp+A0h] [rbp-70h]
  _BYTE v53[32]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v54; // [rsp+D0h] [rbp-40h]

  v6 = a2;
  v7 = *(_QWORD *)(a6 + 16);
  v8 = (16 * a5) & 0xF400FA | 0xB0005;
  v9 = *(_QWORD *)(a2 + 32);
  v10 = *(_QWORD *)(v7 + 16);
  v48 = v50;
  v11 = *(_QWORD *)(v10 + 16);
  v49 = 0x300000000LL;
  sub_94F380(v9, a3, &v46, &v47);
  v12 = sub_92F410(a2, v7);
  v44 = sub_92F410(a2, v10);
  v43 = sub_92F410(a2, v11);
  v13 = sub_BCB2D0(*(_QWORD *)(a2 + 40));
  v14 = sub_ACD640(v13, v8, 0);
  v15 = (unsigned int)v49;
  v16 = (unsigned int)v49 + 1LL;
  if ( v16 > HIDWORD(v49) )
  {
    sub_C8D5F0(&v48, v50, v16, 8);
    v15 = (unsigned int)v49;
  }
  v17 = (unsigned int **)(a2 + 48);
  *(_QWORD *)&v48[8 * v15] = v14;
  v52 = 257;
  v18 = (unsigned int)(v49 + 1);
  LODWORD(v49) = v49 + 1;
  if ( v47 == v12->m128i_i64[1] )
    goto LABEL_20;
  if ( v12->m128i_i8[0] <= 0x15u )
  {
    v19 = *(_QWORD *)(a2 + 128);
    v20 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v19 + 136LL);
    if ( v20 == sub_928970 )
      v12 = (__m128i *)sub_ADAFB0(v12, v47);
    else
      v12 = (__m128i *)v20(v19, (__int64)v12, v47);
    if ( v12->m128i_i8[0] > 0x1Cu )
    {
      (*(void (__fastcall **)(_QWORD, __m128i *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
        *(_QWORD *)(a2 + 136),
        v12,
        v51,
        *(_QWORD *)(a2 + 104),
        *(_QWORD *)(a2 + 112));
      v21 = *(unsigned int **)(a2 + 48);
      v22 = &v21[4 * *(unsigned int *)(a2 + 56)];
      while ( v22 != v21 )
      {
        v23 = *((_QWORD *)v21 + 1);
        v24 = *v21;
        v21 += 4;
        sub_B99FD0(v12, v24, v23);
      }
      v18 = (unsigned int)v49;
      goto LABEL_11;
    }
    v18 = (unsigned int)v49;
LABEL_20:
    v25 = v18 + 1;
    if ( v18 + 1 <= (unsigned __int64)HIDWORD(v49) )
      goto LABEL_12;
    goto LABEL_21;
  }
  v54 = 257;
  v12 = (__m128i *)sub_B52210(v12, v47, v53, 0, 0);
  (*(void (__fastcall **)(_QWORD, __m128i *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
    *(_QWORD *)(a2 + 136),
    v12,
    v51,
    *(_QWORD *)(a2 + 104),
    *(_QWORD *)(a2 + 112));
  v35 = *(_QWORD *)(a2 + 48);
  v36 = 16LL * *(unsigned int *)(a2 + 56);
  if ( v35 != v35 + v36 )
  {
    v37 = (unsigned int *)(v35 + v36);
    v38 = *(unsigned int **)(a2 + 48);
    do
    {
      v39 = *((_QWORD *)v38 + 1);
      v40 = *v38;
      v38 += 4;
      sub_B99FD0(v12, v40, v39);
    }
    while ( v37 != v38 );
    v6 = a2;
  }
  v18 = (unsigned int)v49;
LABEL_11:
  v25 = v18 + 1;
  if ( v18 + 1 <= (unsigned __int64)HIDWORD(v49) )
    goto LABEL_12;
LABEL_21:
  sub_C8D5F0(&v48, v50, v25, 8);
  v18 = (unsigned int)v49;
LABEL_12:
  *(_QWORD *)&v48[8 * v18] = v12;
  LODWORD(v49) = v49 + 1;
  v26 = sub_94B510(v6, v46, (__int64)v44);
  v27 = (unsigned int)v49;
  v28 = (unsigned int)v49 + 1LL;
  if ( v28 > HIDWORD(v49) )
  {
    sub_C8D5F0(&v48, v50, v28, 8);
    v27 = (unsigned int)v49;
  }
  *(_QWORD *)&v48[8 * v27] = v26;
  v29 = *(__int64 **)(v6 + 32);
  LODWORD(v49) = v49 + 1;
  v51[0] = v47;
  v30 = sub_90A810(v29, a4, (__int64)v51, 1u);
  v31 = 0;
  v54 = 257;
  if ( v30 )
    v31 = *(_QWORD *)(v30 + 24);
  v32 = sub_921880(v17, v31, v30, (int)v48, v49, (__int64)v53, 0);
  sub_94B940(v6, v32, (__int64)v43);
  v33 = v48;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  if ( v33 != v50 )
    _libc_free(v33, v32);
  return a1;
}
