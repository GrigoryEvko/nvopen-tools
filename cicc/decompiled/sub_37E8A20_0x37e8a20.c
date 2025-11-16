// Function: sub_37E8A20
// Address: 0x37e8a20
//
void __fastcall sub_37E8A20(__m128i *a1, __int64 a2)
{
  __m128i *v2; // r12
  int v3; // ebx
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 (*v6)(void); // rdx
  __int64 v7; // r15
  __int64 v8; // r13
  __m128i *v9; // rsi
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rcx
  __int64 *v14; // rbx
  __int64 v15; // rdx
  unsigned __int64 *v16; // r14
  const __m128i *v17; // r12
  const __m128i *i; // r15
  const __m128i *v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rsi
  __int64 v24; // r14
  __int64 v25; // r10
  __int64 v26; // rdx
  void (__fastcall *v27)(__int64, __int64, __int64, __int64, __int64 *, unsigned __int64, __int64); // rbx
  __int64 v28; // rax
  unsigned __int64 v29; // r9
  unsigned __int64 v30; // rbx
  __int64 v31; // rdx
  unsigned __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rdi
  int v36; // eax
  unsigned __int64 *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 (*v40)(); // rax
  __int64 v41; // rax
  __int64 v42; // rdi
  __int64 v43; // rbx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  int v47; // eax
  __int64 v48; // r12
  unsigned __int64 *v49; // rcx
  unsigned __int64 v50; // rdx
  __m128i *v51; // [rsp+0h] [rbp-160h]
  unsigned int v52; // [rsp+8h] [rbp-158h]
  unsigned int v53; // [rsp+Ch] [rbp-154h]
  __int64 v54; // [rsp+10h] [rbp-150h]
  __int64 v55; // [rsp+18h] [rbp-148h]
  __int64 v57; // [rsp+20h] [rbp-140h]
  __m128i *v58; // [rsp+20h] [rbp-140h]
  __int64 *v59; // [rsp+28h] [rbp-138h]
  __int64 v60; // [rsp+28h] [rbp-138h]
  __int64 v61; // [rsp+38h] [rbp-128h] BYREF
  __int64 v62; // [rsp+40h] [rbp-120h] BYREF
  __int64 v63; // [rsp+48h] [rbp-118h]
  __int64 v64[8]; // [rsp+50h] [rbp-110h] BYREF
  char v65; // [rsp+90h] [rbp-D0h] BYREF

  v2 = a1;
  v3 = -1;
  v4 = a1[32].m128i_i64[1];
  v55 = *(_QWORD *)(a2 + 24);
  v64[6] = (__int64)&v65;
  v64[7] = 0x400000000LL;
  v5 = *(_QWORD *)v4;
  v6 = *(__int64 (**)(void))(*(_QWORD *)v4 + 168LL);
  if ( v6 != sub_2E77FD0 )
  {
    v47 = v6();
    v4 = v2[32].m128i_i64[1];
    v3 = v47;
    v5 = *(_QWORD *)v4;
  }
  v7 = (*(__int64 (__fastcall **)(__int64, __int64))(v5 + 328))(v4, a2);
  v53 = *(_DWORD *)(v2[12].m128i_i64[1] + 8LL * *(int *)(v7 + 24));
  v52 = sub_37E6C00((__int64)v2, a2);
  v8 = v55;
  *(_DWORD *)(v2[12].m128i_i64[1] + 8LL * *(int *)(v55 + 24) + 4) -= v3;
  if ( v55 + 48 != (*(_QWORD *)(v55 + 48) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v9 = (__m128i *)v55;
    v10 = sub_37E6EE0((__int64)v2, v55, *(_QWORD *)(v55 + 16));
    v13 = v55;
    v14 = *(__int64 **)(v55 + 112);
    v8 = v10;
    v15 = v10 + 184;
    v59 = &v14[*(unsigned int *)(v55 + 120)];
    if ( v14 != v59 )
    {
      v54 = v7;
      v16 = (unsigned __int64 *)(v10 + 184);
      v51 = v2;
      do
      {
        v17 = *(const __m128i **)(*v14 + 192);
        for ( i = (const __m128i *)sub_2E33140(*v14); v17 != i; *(_QWORD *)(v8 + 192) = v9 )
        {
          while ( 1 )
          {
            v9 = *(__m128i **)(v8 + 192);
            if ( v9 != *(__m128i **)(v8 + 200) )
              break;
            v19 = i;
            i = (const __m128i *)((char *)i + 24);
            sub_2E33890(v16, v9, v19);
            if ( v17 == i )
              goto LABEL_13;
          }
          if ( v9 )
          {
            *v9 = _mm_loadu_si128(i);
            v9[1].m128i_i64[0] = i[1].m128i_i64[0];
            v9 = *(__m128i **)(v8 + 192);
          }
          v9 = (__m128i *)((char *)v9 + 24);
          i = (const __m128i *)((char *)i + 24);
        }
LABEL_13:
        ++v14;
      }
      while ( v59 != v14 );
      v7 = v54;
      v2 = v51;
    }
    sub_2E31EE0(v8, (__int64)v9, v15, v13, v11, v12);
    sub_2E33F80(v8, v7, -1, v20, v21, v22);
    sub_2E33690(v55, v7, v8);
    if ( v55 == v2[21].m128i_i64[1] )
      v2[21].m128i_i64[1] = v8;
  }
  v23 = *(_QWORD *)(a2 + 56);
  v61 = v23;
  if ( v23 )
    sub_B96E90((__int64)&v61, v23, 1);
  sub_2E88E20(a2);
  v24 = sub_37E6EE0((__int64)v2, *(_QWORD *)(v2[31].m128i_i64[1] + 320) & 0xFFFFFFFFFFFFFFF8LL, *(_QWORD *)(v7 + 16));
  *(_BYTE *)((*(_QWORD *)v24 & 0xFFFFFFFFFFFFFFF8LL) + 261) = *(_BYTE *)(v24 + 261);
  *(_BYTE *)(v24 + 261) = 0;
  v25 = v2[32].m128i_i64[1];
  v26 = v2[27].m128i_i64[0];
  v27 = *(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64 *, unsigned __int64, __int64))(*(_QWORD *)v25 + 336LL);
  if ( *(_DWORD *)(v7 + 256) == *(_DWORD *)(v8 + 256) && *(_DWORD *)(v7 + 252) == *(_DWORD *)(v8 + 252) )
  {
    v29 = v53 - (unsigned __int64)v52;
  }
  else
  {
    v57 = v2[27].m128i_i64[0];
    v60 = v2[32].m128i_i64[1];
    v28 = sub_23CF1D0(v2[33].m128i_i64[0]);
    v26 = v57;
    v25 = v60;
    v29 = v28;
  }
  v27(v25, v8, v7, v24, &v61, v29, v26);
  *(_DWORD *)(v2[12].m128i_i64[1] + 8LL * *(int *)(v8 + 24) + 4) = sub_37E70D0((__int64)v2, v8);
  sub_37E6D50((__int64)v2, v55, *(_QWORD *)(v8 + 8));
  v58 = v2 + 22;
  if ( v24 + 48 == (*(_QWORD *)(v24 + 48) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v48 = v2[31].m128i_i64[1] + 320;
    sub_2E31020(v48, v24);
    v49 = *(unsigned __int64 **)(v24 + 8);
    v50 = *(_QWORD *)v24 & 0xFFFFFFFFFFFFFFF8LL;
    *v49 = v50 | *v49 & 7;
    *(_QWORD *)(v50 + 8) = v49;
    *(_QWORD *)v24 &= 7uLL;
    *(_QWORD *)(v24 + 8) = 0;
    sub_2E79D60(v48, (_QWORD *)v24);
    v62 = v8;
    v63 = v7;
    sub_37E85E0((__int64)v64, v58, &v62);
  }
  else
  {
    if ( *(_DWORD *)(v55 + 252) == unk_501EB38
      && *(_DWORD *)(v55 + 256) == unk_501EB3C
      && (*(_DWORD *)(v7 + 256) != unk_501EB3C || *(_DWORD *)(v7 + 252) != unk_501EB38) )
    {
      v41 = sub_37E6EE0((__int64)v2, v2[21].m128i_i64[1], *(_QWORD *)(v2[21].m128i_i64[1] + 16));
      v42 = v2[32].m128i_i64[1];
      v43 = v41;
      v64[0] = 0;
      (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD, _QWORD, __int64 *))(*(_QWORD *)v42 + 368LL))(
        v42,
        v41,
        v7,
        0,
        0,
        0,
        v64);
      if ( v64[0] )
        sub_B91220((__int64)v64, v64[0]);
      *(_DWORD *)(v2[12].m128i_i64[1] + 8LL * *(int *)(v43 + 24) + 4) = sub_37E70D0((__int64)v2, v43);
      sub_37E6D50((__int64)v2, v2[21].m128i_i64[1], *(_QWORD *)(v43 + 8));
      v2[21].m128i_i64[1] = v43;
      sub_2E33690(v8, v7, v43);
      sub_2E33F80(v43, v7, -1, v44, v45, v46);
      v7 = v43;
    }
    v30 = *(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL;
    v31 = sub_2E32300((__int64 *)v30, 0);
    if ( v31 )
    {
      v35 = v2[32].m128i_i64[1];
      v64[0] = 0;
      (*(void (__fastcall **)(__int64, unsigned __int64, __int64, _QWORD, _QWORD, _QWORD, __int64 *, _QWORD))(*(_QWORD *)v35 + 368LL))(
        v35,
        v30,
        v31,
        0,
        0,
        0,
        v64,
        0);
      if ( v64[0] )
        sub_B91220((__int64)v64, v64[0]);
      v36 = sub_37E70D0((__int64)v2, v30);
      v32 = v2[12].m128i_u64[1];
      *(_DWORD *)(v32 + 8LL * *(int *)(v30 + 24) + 4) = v36;
    }
    v37 = *(unsigned __int64 **)(v24 + 8);
    if ( (unsigned __int64 *)v7 != v37 && v24 != v7 && (unsigned __int64 *)v7 != v37 && (unsigned __int64 *)v24 != v37 )
    {
      v32 = *v37 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)((*(_QWORD *)v24 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v37;
      *v37 = *v37 & 7 | *(_QWORD *)v24 & 0xFFFFFFFFFFFFFFF8LL;
      v38 = *(_QWORD *)v7;
      *(_QWORD *)(v32 + 8) = v7;
      v38 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v24 = v38 | *(_QWORD *)v24 & 7LL;
      *(_QWORD *)(v38 + 8) = v24;
      *(_QWORD *)v7 = v32 | *(_QWORD *)v7 & 7LL;
    }
    sub_2E33F80(v24, v7, -1, v32, v33, v34);
    sub_2E33690(v8, v7, v24);
    v39 = v2[32].m128i_i64[0];
    v40 = *(__int64 (**)())(*(_QWORD *)v39 + 528LL);
    if ( v40 == sub_2FF52D0 || ((unsigned __int8 (__fastcall *)(__int64, __int64))v40)(v39, v2[31].m128i_i64[1]) )
      sub_3509790(&v2[27].m128i_i64[1], (_QWORD *)v24);
    *(_DWORD *)(v2[12].m128i_i64[1] + 8LL * *(int *)(v24 + 24) + 4) = sub_37E70D0((__int64)v2, v24);
    sub_37E6D50((__int64)v2, v30, v7);
    *(_QWORD *)(v24 + 252) = *(_QWORD *)(v7 + 252);
    *(_BYTE *)(v24 + 260) = *(_BYTE *)(v7 + 260);
    *(_BYTE *)(v7 + 260) = 0;
    v62 = v8;
    v63 = v24;
    sub_37E85E0((__int64)v64, v58, &v62);
  }
  if ( v61 )
    sub_B91220((__int64)&v61, v61);
}
