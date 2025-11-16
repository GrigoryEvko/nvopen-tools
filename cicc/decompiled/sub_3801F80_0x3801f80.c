// Function: sub_3801F80
// Address: 0x3801f80
//
__int64 *__fastcall sub_3801F80(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __m128i v5; // xmm0
  __m128i v6; // xmm1
  __int64 v7; // rax
  unsigned int v8; // esi
  _QWORD *v9; // r13
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v14; // rsi
  __int64 v15; // r14
  __int64 v16; // r8
  unsigned int v17; // r15d
  int v18; // edx
  __int64 v19; // [rsp+8h] [rbp-98h]
  unsigned int v20; // [rsp+2Ch] [rbp-74h] BYREF
  __int128 v21; // [rsp+30h] [rbp-70h] BYREF
  __int128 v22; // [rsp+40h] [rbp-60h] BYREF
  __int64 v23; // [rsp+50h] [rbp-50h] BYREF
  int v24; // [rsp+58h] [rbp-48h]
  __int64 v25; // [rsp+60h] [rbp-40h] BYREF
  int v26; // [rsp+68h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(a2 + 80);
  v23 = 0;
  v24 = 0;
  v5 = _mm_loadu_si128((const __m128i *)(v3 + 80));
  v6 = _mm_loadu_si128((const __m128i *)(v3 + 120));
  v7 = *(_QWORD *)(v3 + 40);
  v21 = (__int128)v5;
  LODWORD(v7) = *(_DWORD *)(v7 + 96);
  v22 = (__int128)v6;
  v25 = v4;
  v20 = v7;
  if ( v4 )
    sub_B96E90((__int64)&v25, v4, 1);
  v26 = *(_DWORD *)(a2 + 72);
  sub_38014E0(a1, (unsigned __int64 *)&v21, (__int64)&v22, &v20, (__int64)&v25, (__int64)&v23, 0);
  if ( v25 )
    sub_B91220((__int64)&v25, v25);
  if ( (_QWORD)v22 )
  {
    v8 = v20;
  }
  else
  {
    v14 = *(_QWORD *)(a2 + 80);
    v15 = a1[1];
    v16 = *(_QWORD *)(*(_QWORD *)(v21 + 48) + 16LL * DWORD2(v21) + 8);
    v17 = *(unsigned __int16 *)(*(_QWORD *)(v21 + 48) + 16LL * DWORD2(v21));
    v25 = v14;
    if ( v14 )
    {
      v19 = v16;
      sub_B96E90((__int64)&v25, v14, 1);
      v16 = v19;
    }
    v26 = *(_DWORD *)(a2 + 72);
    *(_QWORD *)&v22 = sub_3400BD0(v15, 0, (__int64)&v25, v17, v16, 0, v5, 0);
    DWORD2(v22) = v18;
    if ( v25 )
      sub_B91220((__int64)&v25, v25);
    v20 = 22;
    v8 = 22;
  }
  v9 = (_QWORD *)a1[1];
  v10 = *(_QWORD *)(a2 + 40);
  v11 = sub_33ED040(v9, v8);
  return sub_33EC430(
           v9,
           (__int64 *)a2,
           **(_QWORD **)(a2 + 40),
           *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
           v11,
           v12,
           v21,
           v22,
           *(_OWORD *)(v10 + 160));
}
