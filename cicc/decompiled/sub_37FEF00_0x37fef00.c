// Function: sub_37FEF00
// Address: 0x37fef00
//
void __fastcall sub_37FEF00(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  __int64 v8; // r9
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v10; // rax
  unsigned __int16 v11; // si
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // r9
  __int64 v15; // rsi
  __int64 v16; // rdx
  int v17; // eax
  int v18; // edx
  void *v19; // rax
  void *v20; // rbx
  int v21; // edx
  __int64 v22; // rax
  __int64 v23; // rax
  _QWORD *v24; // rdi
  __m128i v25; // xmm2
  __m128i v26; // xmm1
  int v27; // edx
  _QWORD *i; // r14
  __int64 v29; // rdx
  __int128 v30; // [rsp-10h] [rbp-100h]
  __int64 v31; // [rsp-10h] [rbp-100h]
  __int64 v32; // [rsp+0h] [rbp-F0h]
  unsigned __int64 v33; // [rsp+8h] [rbp-E8h]
  void *v34; // [rsp+10h] [rbp-E0h]
  __int64 v36; // [rsp+20h] [rbp-D0h]
  char v37; // [rsp+2Fh] [rbp-C1h]
  unsigned __int8 *v38; // [rsp+40h] [rbp-B0h]
  __m128i v39; // [rsp+60h] [rbp-90h] BYREF
  __int64 v40; // [rsp+70h] [rbp-80h] BYREF
  int v41; // [rsp+78h] [rbp-78h]
  __m128i v42; // [rsp+80h] [rbp-70h] BYREF
  __int16 v43; // [rsp+90h] [rbp-60h]
  __int64 v44; // [rsp+98h] [rbp-58h]
  __m128i v45; // [rsp+A0h] [rbp-50h] BYREF
  __m128i v46; // [rsp+B0h] [rbp-40h]

  v8 = *a1;
  v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v10 = *(__int16 **)(a2 + 48);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  v13 = a1[1];
  if ( v9 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v45, v8, *(_QWORD *)(v13 + 64), v11, v12);
    v39.m128i_i16[0] = v45.m128i_i16[4];
    v39.m128i_i64[1] = v46.m128i_i64[0];
  }
  else
  {
    v39.m128i_i32[0] = v9(v8, *(_QWORD *)(v13 + 64), v11, v12);
    v39.m128i_i64[1] = v29;
  }
  v15 = *(_QWORD *)(a2 + 80);
  v40 = v15;
  if ( v15 )
    sub_B96E90((__int64)&v40, v15, 1);
  v16 = *(_QWORD *)(a2 + 40);
  v41 = *(_DWORD *)(a2 + 72);
  v17 = *(_DWORD *)(a2 + 24);
  if ( v17 > 239 )
  {
    if ( (unsigned int)(v17 - 242) > 1 )
    {
LABEL_8:
      v15 = 233;
      v33 = 0;
      v32 = 0;
      v37 = 0;
      *(_QWORD *)a4 = sub_33FAF80(a1[1], 233, (__int64)&v40, v39.m128i_u32[0], v39.m128i_i64[1], v14, a5);
      *(_DWORD *)(a4 + 8) = v18;
      goto LABEL_9;
    }
  }
  else if ( v17 <= 237 && (unsigned int)(v17 - 101) > 0x2F )
  {
    goto LABEL_8;
  }
  v22 = *(_QWORD *)(*(_QWORD *)(v16 + 40) + 48LL) + 16LL * *(unsigned int *)(v16 + 48);
  if ( v39.m128i_i16[0] == *(_WORD *)v22 && (v39.m128i_i16[0] || v39.m128i_i64[1] == *(_QWORD *)(v22 + 8)) )
  {
    v37 = 1;
    *(_QWORD *)a4 = *(_QWORD *)(v16 + 40);
    *(_DWORD *)(a4 + 8) = *(_DWORD *)(v16 + 48);
    v23 = *(_QWORD *)(a2 + 40);
    v33 = *(_QWORD *)v23;
    v32 = *(unsigned int *)(v23 + 8);
  }
  else
  {
    a5 = _mm_loadu_si128((const __m128i *)v16);
    v24 = (_QWORD *)a1[1];
    v25 = _mm_loadu_si128(&v39);
    v43 = 1;
    v45 = a5;
    v26 = _mm_loadu_si128((const __m128i *)(v16 + 40));
    *((_QWORD *)&v30 + 1) = 2;
    *(_QWORD *)&v30 = &v45;
    v44 = 0;
    v46 = v26;
    v42 = v25;
    v37 = 1;
    v38 = sub_3411BE0(v24, 0x92u, (__int64)&v40, (unsigned __int16 *)&v42, 2, v14, v30);
    *(_QWORD *)a4 = v38;
    v33 = (unsigned __int64)v38;
    *(_DWORD *)(a4 + 8) = v27;
    v15 = v31;
    v32 = 1;
  }
LABEL_9:
  v36 = a1[1];
  v34 = sub_300AC80((unsigned __int16 *)&v39, v15);
  v19 = sub_C33340();
  v20 = v19;
  if ( v34 == v19 )
    sub_C3C500(&v45, (__int64)v19);
  else
    sub_C373C0(&v45, (__int64)v34);
  if ( (void *)v45.m128i_i64[0] == v20 )
    sub_C3CEB0((void **)&v45, 0);
  else
    sub_C37310((__int64)&v45, 0);
  *(_QWORD *)a3 = sub_33FE6E0(v36, v45.m128i_i64, (__int64)&v40, v39.m128i_u32[0], v39.m128i_i64[1], 0, a5);
  *(_DWORD *)(a3 + 8) = v21;
  if ( (void *)v45.m128i_i64[0] == v20 )
  {
    if ( v45.m128i_i64[1] )
    {
      for ( i = (_QWORD *)(v45.m128i_i64[1] + 24LL * *(_QWORD *)(v45.m128i_i64[1] - 8));
            (_QWORD *)v45.m128i_i64[1] != i;
            sub_91D830(i) )
      {
        i -= 3;
      }
      j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v45);
  }
  if ( v37 )
    sub_3760E70((__int64)a1, a2, 1, v33, v32);
  if ( v40 )
    sub_B91220((__int64)&v40, v40);
}
