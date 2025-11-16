// Function: sub_36EB6A0
// Address: 0x36eb6a0
//
void __fastcall sub_36EB6A0(__int64 a1, __int64 a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v9; // rdx
  _QWORD *v10; // rax
  __int64 v11; // rsi
  char v12; // bl
  int v13; // edx
  char v14; // al
  _QWORD *v15; // rdi
  unsigned __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r15
  _QWORD *v23; // rdi
  unsigned __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // rax
  _QWORD *v27; // rbx
  __int64 v28; // rsi
  __int64 v29; // rdi
  __int64 v30; // r9
  int v31; // edx
  _QWORD *v32; // rdi
  unsigned __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // [rsp+0h] [rbp-70h] BYREF
  int v36; // [rsp+8h] [rbp-68h]
  __int64 v37; // [rsp+10h] [rbp-60h] BYREF
  int v38; // [rsp+18h] [rbp-58h]
  _OWORD v39[5]; // [rsp+20h] [rbp-50h] BYREF

  v9 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL) + 96LL);
  v10 = *(_QWORD **)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v10 = (_QWORD *)*v10;
  v11 = *(_QWORD *)(a2 + 80);
  v12 = (char)v10;
  v35 = v11;
  if ( v11 )
    sub_B96E90((__int64)&v35, v11, 1);
  v13 = *(_DWORD *)(a2 + 72);
  v14 = v12 & 0xF;
  v36 = v13;
  if ( (v12 & 0xF) == 1 )
  {
    v26 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 120LL) + 96LL);
    v27 = *(_QWORD **)(v26 + 24);
    if ( *(_DWORD *)(v26 + 32) > 0x40u )
      v27 = (_QWORD *)*v27;
    v28 = *(_QWORD *)(a2 + 80);
    v37 = v28;
    if ( v28 )
    {
      sub_B96E90((__int64)&v37, v28, 1);
      v13 = *(_DWORD *)(a2 + 72);
    }
    v29 = *(_QWORD *)(a1 + 64);
    v38 = v13;
    *(_QWORD *)&v39[0] = sub_3400BD0(v29, (unsigned int)v27, (__int64)&v37, 7, 0, 1u, a3, 0);
    DWORD2(v39[0]) = v31;
    if ( v37 )
      sub_B91220((__int64)&v37, v37);
    v32 = *(_QWORD **)(a1 + 64);
    v33 = *(_QWORD *)(a2 + 48);
    v34 = *(unsigned int *)(a2 + 68);
    v39[1] = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
    v18 = sub_33E66D0(v32, 2873, (__int64)&v35, v33, v34, v30, (unsigned __int64 *)v39, 2);
  }
  else if ( v14 == 2 )
  {
    v15 = *(_QWORD **)(a1 + 64);
    v16 = *(_QWORD *)(a2 + 48);
    v17 = *(unsigned int *)(a2 + 68);
    v39[0] = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
    v18 = sub_33E66D0(v15, 2872, (__int64)&v35, v16, v17, a7, (unsigned __int64 *)v39, 1);
  }
  else
  {
    if ( v14 )
      BUG();
    v23 = *(_QWORD **)(a1 + 64);
    v24 = *(_QWORD *)(a2 + 48);
    v25 = *(unsigned int *)(a2 + 68);
    v39[0] = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
    v18 = sub_33E66D0(v23, 2871, (__int64)&v35, v24, v25, a7, (unsigned __int64 *)v39, 1);
  }
  v22 = v18;
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v18, v19, v20, v21);
  sub_3421DB0(v22);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v35 )
    sub_B91220((__int64)&v35, v35);
}
