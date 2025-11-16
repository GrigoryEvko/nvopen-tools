// Function: sub_2080770
// Address: 0x2080770
//
__int64 *__fastcall sub_2080770(__int64 a1, __int64 a2, int a3, __m128i a4, __m128i a5, __m128i a6)
{
  __int64 *v8; // r12
  __int64 v9; // rdx
  int v10; // ebx
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // eax
  unsigned int v17; // edx
  int v18; // edx
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rax
  const void **v23; // rdx
  const void **v24; // r8
  __int64 v25; // rcx
  char v26; // al
  int v27; // eax
  int v28; // edx
  __int64 *v29; // rdi
  __int64 v30; // r13
  int v31; // edx
  int v32; // r12d
  __int64 *result; // rax
  __int64 v34; // rsi
  int v35; // edx
  int v36; // edx
  int v37; // edx
  int v38; // edx
  int v39; // edx
  int v40; // edx
  int v41; // edx
  int v42; // edx
  int v43; // edx
  int v44; // edx
  int v45; // edx
  int v46; // edx
  __int128 v47; // [rsp-10h] [rbp-A0h]
  __int128 v48; // [rsp-10h] [rbp-A0h]
  __int128 v49; // [rsp+0h] [rbp-90h]
  __int64 v50; // [rsp+18h] [rbp-78h]
  int v51; // [rsp+18h] [rbp-78h]
  __int64 v52; // [rsp+18h] [rbp-78h]
  __int64 v53; // [rsp+18h] [rbp-78h]
  __int64 v55; // [rsp+28h] [rbp-68h]
  const void **v56; // [rsp+28h] [rbp-68h]
  __int64 v57; // [rsp+48h] [rbp-48h] BYREF
  __int64 v58; // [rsp+50h] [rbp-40h] BYREF
  int v59; // [rsp+58h] [rbp-38h]

  v8 = 0;
  v55 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 16LL);
  *(_QWORD *)&v49 = sub_20685E0(a1, *(__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), a4, a5, a6);
  *((_QWORD *)&v49 + 1) = v9;
  v10 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v11 = 0;
  if ( *(char *)(a2 + 23) >= 0 )
    goto LABEL_8;
  v12 = sub_1648A40(a2);
  v50 = v13 + v12;
  if ( *(char *)(a2 + 23) >= 0 )
  {
    if ( (unsigned int)(v50 >> 4) )
LABEL_44:
      BUG();
LABEL_8:
    v16 = 0;
    goto LABEL_9;
  }
  if ( !(unsigned int)((v50 - sub_1648A40(a2)) >> 4) )
    goto LABEL_8;
  if ( *(char *)(a2 + 23) >= 0 )
    goto LABEL_44;
  v51 = *(_DWORD *)(sub_1648A40(a2) + 8);
  if ( *(char *)(a2 + 23) >= 0 )
    BUG();
  v14 = sub_1648A40(a2);
  v16 = *(_DWORD *)(v14 + v15 - 4) - v51;
LABEL_9:
  if ( (unsigned int)(v10 - 1 - v16) > 1 )
  {
    v8 = sub_20685E0(a1, *(__int64 **)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))), a4, a5, a6);
    v11 = v17;
  }
  v18 = *(_DWORD *)(a1 + 536);
  v19 = *(_QWORD *)a1;
  v58 = 0;
  v59 = v18;
  if ( v19 )
  {
    if ( &v58 != (__int64 *)(v19 + 48) )
    {
      v20 = *(_QWORD *)(v19 + 48);
      v58 = v20;
      if ( v20 )
        sub_1623A60((__int64)&v58, v20, 2);
    }
  }
  v52 = *(_QWORD *)a2;
  v21 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL));
  LOBYTE(v22) = sub_204D4D0(v55, v21, v52);
  v24 = v23;
  v25 = v22;
  v26 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
  if ( v26 == 16 )
    v26 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL);
  if ( (unsigned __int8)(v26 - 1) <= 5u || (v28 = 0, *(_BYTE *)(a2 + 16) == 76) )
  {
    v53 = v25;
    v56 = v24;
    v27 = sub_15F24E0(a2);
    v25 = v53;
    v24 = v56;
    v28 = v27;
  }
  v29 = *(__int64 **)(a1 + 552);
  switch ( a3 )
  {
    case 'S':
      v30 = sub_1D309E0(
              v29,
              248,
              (__int64)&v58,
              v25,
              v24,
              0,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64,
              *(double *)a6.m128i_i64,
              v49);
      v32 = v36;
      break;
    case 'T':
      v30 = sub_1D309E0(
              v29,
              250,
              (__int64)&v58,
              v25,
              v24,
              0,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64,
              *(double *)a6.m128i_i64,
              v49);
      v32 = v37;
      break;
    case 'U':
      *((_QWORD *)&v47 + 1) = v11;
      *(_QWORD *)&v47 = v8;
      if ( v28 == -1 )
        v30 = sub_1D309E0(
                v29,
                246,
                (__int64)&v58,
                v25,
                v24,
                0,
                *(double *)a4.m128i_i64,
                *(double *)a5.m128i_i64,
                *(double *)a6.m128i_i64,
                v47);
      else
        v30 = (__int64)sub_1D332F0(
                         v29,
                         244,
                         (__int64)&v58,
                         v25,
                         v24,
                         0,
                         *(double *)a4.m128i_i64,
                         *(double *)a5.m128i_i64,
                         a6,
                         v49,
                         *((unsigned __int64 *)&v49 + 1),
                         v47);
      v32 = v38;
      break;
    case 'V':
      v30 = sub_1D309E0(
              v29,
              257,
              (__int64)&v58,
              v25,
              v24,
              0,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64,
              *(double *)a6.m128i_i64,
              v49);
      v32 = v39;
      break;
    case 'W':
      v30 = sub_1D309E0(
              v29,
              258,
              (__int64)&v58,
              v25,
              v24,
              0,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64,
              *(double *)a6.m128i_i64,
              v49);
      v32 = v40;
      break;
    case 'X':
      *((_QWORD *)&v48 + 1) = v11;
      *(_QWORD *)&v48 = v8;
      if ( v28 == -1 )
        v30 = sub_1D309E0(
                v29,
                247,
                (__int64)&v58,
                v25,
                v24,
                0,
                *(double *)a4.m128i_i64,
                *(double *)a5.m128i_i64,
                *(double *)a6.m128i_i64,
                v48);
      else
        v30 = (__int64)sub_1D332F0(
                         v29,
                         245,
                         (__int64)&v58,
                         v25,
                         v24,
                         0,
                         *(double *)a4.m128i_i64,
                         *(double *)a5.m128i_i64,
                         a6,
                         v49,
                         *((unsigned __int64 *)&v49 + 1),
                         v48);
      v32 = v41;
      break;
    case 'Y':
      v30 = sub_1D309E0(
              v29,
              249,
              (__int64)&v58,
              v25,
              v24,
              0,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64,
              *(double *)a6.m128i_i64,
              v49);
      v32 = v42;
      break;
    case 'Z':
      v30 = sub_1D309E0(
              v29,
              251,
              (__int64)&v58,
              v25,
              v24,
              0,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64,
              *(double *)a6.m128i_i64,
              v49);
      v32 = v43;
      break;
    case '[':
      v30 = sub_1D309E0(
              v29,
              253,
              (__int64)&v58,
              v25,
              v24,
              0,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64,
              *(double *)a6.m128i_i64,
              v49);
      v32 = v44;
      break;
    case '\\':
      v30 = sub_1D309E0(
              v29,
              254,
              (__int64)&v58,
              v25,
              v24,
              0,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64,
              *(double *)a6.m128i_i64,
              v49);
      v32 = v45;
      break;
    case ']':
      v30 = sub_1D309E0(
              v29,
              255,
              (__int64)&v58,
              v25,
              v24,
              0,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64,
              *(double *)a6.m128i_i64,
              v49);
      v32 = v46;
      break;
    case '^':
      v30 = sub_1D309E0(
              v29,
              256,
              (__int64)&v58,
              v25,
              v24,
              0,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64,
              *(double *)a6.m128i_i64,
              v49);
      v32 = v31;
      break;
    case '_':
      v30 = sub_1D309E0(
              v29,
              252,
              (__int64)&v58,
              v25,
              v24,
              0,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64,
              *(double *)a6.m128i_i64,
              v49);
      v32 = v35;
      break;
  }
  v57 = a2;
  result = sub_205F5C0(a1 + 8, &v57);
  v34 = v58;
  result[1] = v30;
  *((_DWORD *)result + 4) = v32;
  if ( v34 )
    return (__int64 *)sub_161E7C0((__int64)&v58, v34);
  return result;
}
