// Function: sub_30478C0
// Address: 0x30478c0
//
__int64 __fastcall sub_30478C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // rdx
  int v8; // eax
  __int64 v9; // rax
  _QWORD *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // r9
  unsigned int v13; // edx
  unsigned int v14; // eax
  __int64 v15; // r15
  _QWORD **v16; // rdi
  __m128i v17; // xmm0
  __int64 v18; // r8
  _QWORD *v19; // rdx
  __int64 v20; // rax
  const __m128i *v21; // rdx
  __m128i v22; // xmm0
  _QWORD *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rdx
  unsigned int v27; // eax
  __int64 v28; // rdx
  bool v29; // cc
  _QWORD *v30; // rdx
  __int64 v31; // rax
  _QWORD *v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rsi
  __int64 v36; // rdx
  _WORD *v37; // rcx
  int v38; // r9d
  __int64 v39; // rbx
  __int64 v40; // rax
  __int64 v41; // rdx
  __m128i v42; // rax
  __int64 v44; // [rsp-10h] [rbp-300h]
  __int64 v45; // [rsp+8h] [rbp-2E8h]
  __m128i v46; // [rsp+10h] [rbp-2E0h] BYREF
  __m128i v47; // [rsp+20h] [rbp-2D0h] BYREF
  __int64 v48; // [rsp+30h] [rbp-2C0h]
  __int64 v49; // [rsp+38h] [rbp-2B8h]
  __int64 v50; // [rsp+40h] [rbp-2B0h] BYREF
  int v51; // [rsp+48h] [rbp-2A8h]
  __int64 v52; // [rsp+50h] [rbp-2A0h] BYREF
  int v53; // [rsp+58h] [rbp-298h]
  __int64 v54; // [rsp+60h] [rbp-290h]
  int v55; // [rsp+68h] [rbp-288h]
  __int64 v56; // [rsp+70h] [rbp-280h]
  int v57; // [rsp+78h] [rbp-278h]
  __int64 v58; // [rsp+80h] [rbp-270h]
  int v59; // [rsp+88h] [rbp-268h]
  __int64 v60; // [rsp+90h] [rbp-260h]
  __int64 v61; // [rsp+98h] [rbp-258h]
  __int64 v62; // [rsp+A0h] [rbp-250h]
  int v63; // [rsp+A8h] [rbp-248h]
  _QWORD *v64; // [rsp+B0h] [rbp-240h] BYREF
  __int64 v65; // [rsp+B8h] [rbp-238h]
  _QWORD v66[70]; // [rsp+C0h] [rbp-230h] BYREF

  v6 = *(_QWORD *)(a2 + 80);
  v50 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v50, v6, 1);
  v51 = *(_DWORD *)(a2 + 72);
  v7 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL);
  v8 = *(_DWORD *)(v7 + 24);
  if ( v8 != 11 && v8 != 35 )
    sub_C64ED0("The first argument of sparse texture intrinsics should be a constant.", 1u);
  v9 = *(_QWORD *)(v7 + 96);
  v10 = *(_QWORD **)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v10 = (_QWORD *)*v10;
  v11 = sub_3400BD0(a4, (_DWORD)v10, (unsigned int)&v50, 7, 0, 1, 0);
  v12 = v44;
  v64 = v66;
  v66[0] = v11;
  v66[1] = v13;
  v65 = 0x2000000001LL;
  v14 = *(_DWORD *)(a2 + 64);
  if ( v14 <= 3 )
  {
    v23 = v66;
    v22 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
    v24 = 2;
  }
  else
  {
    v15 = 160;
    v16 = &v64;
    v17 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 120LL));
    v18 = 40LL * (v14 - 4) + 160;
    v19 = v66;
    v20 = 1;
    while ( 1 )
    {
      *(__m128i *)&v19[2 * v20] = v17;
      v21 = *(const __m128i **)(a2 + 40);
      v20 = (unsigned int)(v65 + 1);
      LODWORD(v65) = v65 + 1;
      if ( v18 == v15 )
        break;
      v17 = _mm_loadu_si128((const __m128i *)((char *)v21 + v15));
      if ( v20 + 1 > (unsigned __int64)HIDWORD(v65) )
      {
        v45 = v18;
        v47.m128i_i64[0] = (__int64)v16;
        v46 = v17;
        sub_C8D5F0((__int64)v16, v66, v20 + 1, 0x10u, v18, v12);
        v20 = (unsigned int)v65;
        v18 = v45;
        v17 = _mm_load_si128(&v46);
        v16 = (_QWORD **)v47.m128i_i64[0];
      }
      v19 = v64;
      v15 += 40;
    }
    v22 = _mm_loadu_si128(v21);
    if ( v20 + 1 > (unsigned __int64)HIDWORD(v65) )
    {
      v47 = v22;
      sub_C8D5F0((__int64)&v64, v66, v20 + 1, 0x10u, v18, v12);
      v23 = v64;
      v22 = _mm_load_si128(&v47);
      v24 = 2LL * (unsigned int)v65;
    }
    else
    {
      v23 = v64;
      v24 = 2 * v20;
    }
  }
  *(__m128i *)&v23[v24] = v22;
  v25 = *(_QWORD *)(a2 + 40);
  v26 = *(_QWORD *)(v25 + 40);
  v27 = v65 + 1;
  LODWORD(v65) = v65 + 1;
  v28 = *(_QWORD *)(v26 + 96);
  v29 = *(_DWORD *)(v28 + 32) <= 0x40u;
  v30 = *(_QWORD **)(v28 + 24);
  if ( !v29 )
    v30 = (_QWORD *)*v30;
  switch ( (_DWORD)v30 )
  {
    case 0x2508:
      v37 = *(_WORD **)(a2 + 48);
      if ( *v37 == 7 )
      {
        v36 = (__int64)v64;
        v38 = 3631;
      }
      else
      {
        if ( *v37 != 12 )
          goto LABEL_42;
        v36 = (__int64)v64;
        v38 = 3628;
      }
      break;
    case 0x2512:
      v37 = *(_WORD **)(a2 + 48);
      if ( *v37 == 7 )
      {
        v36 = (__int64)v64;
        v38 = 3630;
      }
      else
      {
        if ( *v37 != 12 )
          goto LABEL_42;
        v36 = (__int64)v64;
        v38 = 3629;
      }
      break;
    case 0x2528:
      v31 = *(_QWORD *)(*(_QWORD *)(v25 + 360) + 96LL);
      v32 = *(_QWORD **)(v31 + 24);
      if ( *(_DWORD *)(v31 + 32) > 0x40u )
        v32 = (_QWORD *)*v32;
      v33 = sub_3400BD0(a4, (_DWORD)v32, (unsigned int)&v50, 7, 0, 1, 0);
      v35 = v34;
      v36 = (__int64)v64;
      v49 = v35;
      v48 = v33;
      v64[14] = v33;
      *(_DWORD *)(v36 + 120) = v49;
      v37 = *(_WORD **)(a2 + 48);
      if ( *v37 != 7 )
      {
        if ( *v37 == 12 )
        {
          v27 = v65;
          v38 = 3632;
          break;
        }
LABEL_42:
        BUG();
      }
      v27 = v65;
      v38 = 3633;
      break;
    default:
      goto LABEL_42;
  }
  v39 = sub_33E66D0(a4, v38, (unsigned int)&v50, (_DWORD)v37, *(_DWORD *)(a2 + 68), v38, v36, v27);
  v40 = sub_34074A0(a4, &v50, v39, 4, 2, 0);
  v61 = v41;
  v52 = v39;
  v53 = 0;
  v54 = v39;
  v55 = 1;
  v56 = v39;
  v57 = 2;
  v58 = v39;
  v59 = 3;
  v60 = v40;
  v62 = v39;
  v63 = 5;
  v42.m128i_i64[0] = sub_3411660(a4, &v52, 6, &v50);
  if ( v64 != v66 )
  {
    v47 = v42;
    _libc_free((unsigned __int64)v64);
    v42 = v47;
  }
  if ( v50 )
  {
    v47 = v42;
    sub_B91220((__int64)&v50, v50);
    v42.m128i_i64[0] = v47.m128i_i64[0];
  }
  return v42.m128i_i64[0];
}
