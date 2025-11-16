// Function: sub_202AAF0
// Address: 0x202aaf0
//
__int64 *__fastcall sub_202AAF0(__int64 a1, _QWORD *a2, double a3, double a4, __m128i a5)
{
  _QWORD *v5; // rcx
  const void **v7; // rdi
  unsigned int v8; // r14d
  __int64 v9; // rax
  __int64 v10; // rsi
  __int128 v11; // xmm0
  __int64 v12; // r13
  __int64 v13; // r15
  int v14; // edx
  __int64 v15; // rdx
  __int64 v16; // rax
  char v17; // dl
  __int64 v18; // rax
  unsigned int v19; // eax
  __int64 v20; // rdx
  _QWORD *v21; // rsi
  __int64 *v22; // rdi
  __int64 *v23; // r14
  __int128 v25; // rax
  _QWORD *v26; // [rsp+10h] [rbp-80h]
  const void **v27; // [rsp+18h] [rbp-78h]
  __int64 v28; // [rsp+20h] [rbp-70h] BYREF
  int v29; // [rsp+28h] [rbp-68h]
  __int64 v30; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int64 v31; // [rsp+38h] [rbp-58h]
  __int64 v32; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int64 v33; // [rsp+48h] [rbp-48h]
  char v34[8]; // [rsp+50h] [rbp-40h] BYREF
  __int64 v35; // [rsp+58h] [rbp-38h]

  v5 = a2;
  v7 = *(const void ***)(a2[5] + 8LL);
  v8 = *(unsigned __int8 *)a2[5];
  v9 = a2[4];
  v10 = a2[9];
  v27 = v7;
  v11 = (__int128)_mm_loadu_si128((const __m128i *)(v9 + 40));
  v12 = *(_QWORD *)(v9 + 40);
  v28 = v10;
  v13 = *(unsigned int *)(v9 + 48);
  if ( v10 )
  {
    v26 = v5;
    sub_1623A60((__int64)&v28, v10, 2);
    v5 = v26;
    v9 = v26[4];
  }
  v14 = *((_DWORD *)v5 + 16);
  LODWORD(v31) = 0;
  LODWORD(v33) = 0;
  v29 = v14;
  v15 = *(_QWORD *)(v9 + 8);
  v30 = 0;
  v32 = 0;
  sub_2017DE0(a1, *(_QWORD *)v9, v15, &v30, &v32);
  v16 = *(_QWORD *)(v30 + 40) + 16LL * (unsigned int)v31;
  v17 = *(_BYTE *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  v34[0] = v17;
  v35 = v18;
  if ( v17 )
    v19 = word_4305480[(unsigned __int8)(v17 - 14)];
  else
    v19 = sub_1F58D30((__int64)v34);
  v20 = *(_QWORD *)(v12 + 88);
  v21 = *(_QWORD **)(v20 + 24);
  if ( *(_DWORD *)(v20 + 32) > 0x40u )
    v21 = (_QWORD *)*v21;
  v22 = *(__int64 **)(a1 + 8);
  if ( v19 <= (unsigned __int64)v21 )
  {
    *(_QWORD *)&v25 = sub_1D38BB0(
                        (__int64)v22,
                        (__int64)v21 - v19,
                        (__int64)&v28,
                        *(unsigned __int8 *)(*(_QWORD *)(v12 + 40) + 16 * v13),
                        *(const void ***)(*(_QWORD *)(v12 + 40) + 16 * v13 + 8),
                        0,
                        (__m128i)v11,
                        a4,
                        a5,
                        0);
    v23 = sub_1D332F0(v22, 109, (__int64)&v28, v8, v27, 0, *(double *)&v11, a4, a5, v32, v33, v25);
  }
  else
  {
    v23 = sub_1D332F0(v22, 109, (__int64)&v28, v8, v27, 0, *(double *)&v11, a4, a5, v30, v31, v11);
  }
  if ( v28 )
    sub_161E7C0((__int64)&v28, v28);
  return v23;
}
