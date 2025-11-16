// Function: sub_21E0870
// Address: 0x21e0870
//
__int64 __fastcall sub_21E0870(__int64 a1, char a2, __int16 a3, __int64 a4, __m128i a5, double a6, __m128i a7)
{
  __int64 v7; // rsi
  __int64 v8; // r8
  _QWORD *v11; // r13
  int v12; // eax
  __int64 v13; // rcx
  int v14; // edx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rdi
  unsigned int v19; // edx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 *v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // r8
  const __m128i *v28; // r9
  const __m128i *v29; // r14
  __int64 v30; // rcx
  __int64 v31; // r12
  __int64 v33; // [rsp+8h] [rbp-278h]
  __int64 v34; // [rsp+8h] [rbp-278h]
  __int64 v35; // [rsp+8h] [rbp-278h]
  int v36; // [rsp+10h] [rbp-270h]
  const __m128i *v37; // [rsp+10h] [rbp-270h]
  __int64 v38; // [rsp+10h] [rbp-270h]
  __int64 v40; // [rsp+20h] [rbp-260h] BYREF
  int v41; // [rsp+28h] [rbp-258h]
  __int64 v42; // [rsp+30h] [rbp-250h] BYREF
  int v43; // [rsp+38h] [rbp-248h]
  __int64 *v44; // [rsp+40h] [rbp-240h] BYREF
  __int64 v45; // [rsp+48h] [rbp-238h]
  _BYTE v46[560]; // [rsp+50h] [rbp-230h] BYREF

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 32) + 252LL) <= 0x45u )
    sub_16BD130("hmmamma is not supported on this architecture", 1u);
  v7 = *(_QWORD *)(a4 + 72);
  v8 = a1;
  v11 = *(_QWORD **)(a1 - 176);
  v40 = v7;
  if ( v7 )
  {
    sub_1623A60((__int64)&v40, v7, 2);
    v8 = a1;
  }
  v12 = *(_DWORD *)(a4 + 64);
  v13 = *(_QWORD *)(*(_QWORD *)(a4 + 32) + 80LL);
  v41 = v12;
  v14 = *(unsigned __int16 *)(v13 + 24);
  if ( v14 != 10 && v14 != 32 )
    sub_16BD130("rowcol not constant", 1u);
  v15 = *(_QWORD *)(v13 + 88);
  v16 = *(_QWORD *)(v15 + 24);
  if ( *(_DWORD *)(v15 + 32) > 0x40u )
    v16 = **(_QWORD **)(v15 + 24);
  v17 = *(_QWORD *)(a4 + 72);
  v44 = (__int64 *)v46;
  v45 = 0x2000000000LL;
  v42 = v17;
  if ( v17 )
  {
    v33 = v8;
    v36 = v16;
    sub_1623A60((__int64)&v42, v17, 2);
    v12 = *(_DWORD *)(a4 + 64);
    v8 = v33;
    LODWORD(v16) = v36;
  }
  v18 = *(_QWORD *)(v8 - 176);
  v43 = v12;
  v20 = sub_1D38BB0(v18, (unsigned int)v16, (__int64)&v42, 5, 0, 1, a5, a6, a7, 0);
  v21 = v19;
  v22 = (unsigned int)v45;
  if ( (unsigned int)v45 >= HIDWORD(v45) )
  {
    v35 = v20;
    v38 = v19;
    sub_16CD150((__int64)&v44, v46, 0, 16, v20, v19);
    v22 = (unsigned int)v45;
    v20 = v35;
    v21 = v38;
  }
  v23 = &v44[2 * v22];
  *v23 = v20;
  v24 = v42;
  v23[1] = v21;
  v25 = (unsigned int)(v45 + 1);
  LODWORD(v45) = v45 + 1;
  if ( v24 )
  {
    sub_161E7C0((__int64)&v42, v24);
    v25 = (unsigned int)v45;
  }
  v26 = 160;
  v27 = 40LL * (a2 == 0 ? 19 : 23) + 200;
  do
  {
    v28 = (const __m128i *)(v26 + *(_QWORD *)(a4 + 32));
    if ( HIDWORD(v45) <= (unsigned int)v25 )
    {
      v34 = v27;
      v37 = (const __m128i *)(v26 + *(_QWORD *)(a4 + 32));
      sub_16CD150((__int64)&v44, v46, 0, 16, v27, (int)v28);
      v25 = (unsigned int)v45;
      v27 = v34;
      v28 = v37;
    }
    v26 += 40;
    *(__m128i *)&v44[2 * v25] = _mm_loadu_si128(v28);
    v25 = (unsigned int)(v45 + 1);
    LODWORD(v45) = v45 + 1;
  }
  while ( v27 != v26 );
  v29 = *(const __m128i **)(a4 + 32);
  if ( (unsigned int)v25 >= HIDWORD(v45) )
  {
    sub_16CD150((__int64)&v44, v46, 0, 16, v27, (int)v28);
    v25 = (unsigned int)v45;
  }
  *(__m128i *)&v44[2 * v25] = _mm_loadu_si128(v29);
  v30 = *(_QWORD *)(a4 + 40);
  LODWORD(v45) = v45 + 1;
  v31 = sub_1D23DE0(v11, a3, (__int64)&v40, v30, *(_DWORD *)(a4 + 60), (__int64)v28, v44, (unsigned int)v45);
  if ( v44 != (__int64 *)v46 )
    _libc_free((unsigned __int64)v44);
  if ( v40 )
    sub_161E7C0((__int64)&v40, v40);
  return v31;
}
