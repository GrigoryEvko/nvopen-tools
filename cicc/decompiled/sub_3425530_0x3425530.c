// Function: sub_3425530
// Address: 0x3425530
//
void __fastcall sub_3425530(__int64 *a1, __int64 a2)
{
  __int64 v4; // rsi
  int v5; // eax
  const __m128i *v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rax
  const __m128i *v9; // r14
  unsigned __int64 v10; // r15
  __int64 v11; // rax
  unsigned __int64 v12; // rsi
  __m128i *v13; // rdx
  const __m128i *v14; // rax
  _QWORD *v15; // rdi
  unsigned int v16; // esi
  __int64 v17; // r9
  unsigned __int8 *v18; // rax
  __int64 v19; // r14
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int128 v23; // [rsp-10h] [rbp-A0h]
  const __m128i *v24; // [rsp+8h] [rbp-88h]
  __int64 v25; // [rsp+10h] [rbp-80h] BYREF
  int v26; // [rsp+18h] [rbp-78h]
  __int64 v27; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v28; // [rsp+28h] [rbp-68h]
  unsigned __int64 v29; // [rsp+30h] [rbp-60h]
  unsigned __int16 v30; // [rsp+40h] [rbp-50h] BYREF
  __int64 v31; // [rsp+48h] [rbp-48h]
  __int16 v32; // [rsp+50h] [rbp-40h]
  __int64 v33; // [rsp+58h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 80);
  v25 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v25, v4, 1);
  v5 = *(_DWORD *)(a2 + 72);
  v6 = *(const __m128i **)(a2 + 40);
  v27 = 0;
  v28 = 0;
  v26 = v5;
  v7 = *(unsigned int *)(a2 + 64);
  v29 = 0;
  v8 = 40 * v7;
  v9 = (const __m128i *)((char *)v6 + v8);
  v10 = 0xCCCCCCCCCCCCCCD0LL * (v8 >> 3);
  if ( v8 )
  {
    v24 = v6;
    v11 = sub_22077B0(0xCCCCCCCCCCCCCCD0LL * (v8 >> 3));
    v27 = v11;
    v12 = v11;
    v29 = v11 + v10;
    if ( v24 != v9 )
    {
      v13 = (__m128i *)v11;
      v14 = v24;
      do
      {
        if ( v13 )
          *v13 = _mm_loadu_si128(v14);
        v14 = (const __m128i *)((char *)v14 + 40);
        ++v13;
      }
      while ( v14 != v9 );
      v12 = v12 - 0x3333333333333330LL * ((unsigned __int64)((char *)v14 - (char *)v24 - 40) >> 3) + 16;
    }
  }
  else
  {
    v29 = 0;
    v12 = 0;
  }
  v28 = v12;
  sub_3424A90(a1, (unsigned __int64 *)&v27, (__int64)&v25);
  v30 = 1;
  v15 = (_QWORD *)a1[8];
  v32 = 262;
  v16 = *(_DWORD *)(a2 + 24);
  v31 = 0;
  v33 = 0;
  *((_QWORD *)&v23 + 1) = (__int64)(v28 - v27) >> 4;
  *(_QWORD *)&v23 = v27;
  v18 = sub_3411BE0(v15, v16, (__int64)&v25, &v30, 2, v17, v23);
  *((_DWORD *)v18 + 9) = -1;
  v19 = (__int64)v18;
  sub_34158F0(a1[8], a2, (__int64)v18, v20, v21, v22);
  sub_3421DB0(v19);
  sub_33ECEA0((const __m128i *)a1[8], a2);
  if ( v27 )
    j_j___libc_free_0(v27);
  if ( v25 )
    sub_B91220((__int64)&v25, v25);
}
