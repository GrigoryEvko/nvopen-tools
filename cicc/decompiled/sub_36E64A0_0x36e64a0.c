// Function: sub_36E64A0
// Address: 0x36e64a0
//
void __fastcall sub_36E64A0(__int64 a1, __int64 a2, char a3, __m128i a4)
{
  __int64 v7; // rsi
  const __m128i *v8; // rsi
  __int64 v9; // rax
  _QWORD *v10; // rbx
  __m128i v11; // xmm1
  __int64 v12; // rax
  _QWORD *v13; // rsi
  unsigned __int8 *v14; // r8
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r9
  unsigned __int64 v18; // rdx
  unsigned __int64 *v19; // rax
  unsigned int v20; // r15d
  unsigned __int64 v21; // r9
  __int64 v22; // rdi
  __int64 v23; // rax
  unsigned __int64 v24; // r8
  unsigned __int64 v25; // rcx
  __int64 v26; // rdx
  unsigned __int64 **v27; // rdi
  __m128i v28; // xmm0
  __int64 v29; // rax
  _QWORD *v30; // r15
  __m128i v31; // xmm0
  _QWORD *v32; // rdi
  __int64 v33; // r8
  int v34; // esi
  __int64 v35; // r15
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __m128i v39; // [rsp+0h] [rbp-870h] BYREF
  __m128i v40; // [rsp+10h] [rbp-860h] BYREF
  __int64 v41; // [rsp+20h] [rbp-850h] BYREF
  int v42; // [rsp+28h] [rbp-848h]
  unsigned __int64 *v43; // [rsp+30h] [rbp-840h] BYREF
  __int64 v44; // [rsp+38h] [rbp-838h]
  _OWORD v45[131]; // [rsp+40h] [rbp-830h] BYREF

  v7 = *(_QWORD *)(a2 + 80);
  v41 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v41, v7, 1);
  v8 = *(const __m128i **)(a2 + 40);
  v42 = *(_DWORD *)(a2 + 72);
  v9 = *(_QWORD *)(v8[2].m128i_i64[1] + 96);
  v10 = *(_QWORD **)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v10 = (_QWORD *)*v10;
  v11 = _mm_loadu_si128(v8 + 5);
  v43 = (unsigned __int64 *)v45;
  v44 = 0x8000000001LL;
  v45[0] = v11;
  if ( a3 )
  {
    v12 = *(_QWORD *)(v8[7].m128i_i64[1] + 96);
    v13 = *(_QWORD **)(v12 + 24);
    if ( *(_DWORD *)(v12 + 32) > 0x40u )
      v13 = (_QWORD *)*v13;
    v14 = sub_3400BD0(*(_QWORD *)(a1 + 64), (__int64)v13, (__int64)&v41, 7, 0, 1u, a4, 0);
    v15 = (unsigned int)v44;
    v17 = v16;
    v18 = (unsigned int)v44 + 1LL;
    if ( v18 > HIDWORD(v44) )
    {
      v40.m128i_i64[0] = (__int64)v14;
      v40.m128i_i64[1] = v17;
      sub_C8D5F0((__int64)&v43, v45, v18, 0x10u, (__int64)v14, v17);
      v15 = (unsigned int)v44;
      v17 = v40.m128i_i64[1];
      v14 = (unsigned __int8 *)v40.m128i_i64[0];
    }
    v19 = &v43[2 * v15];
    v20 = 4;
    *v19 = (unsigned __int64)v14;
    v19[1] = v17;
    v21 = HIDWORD(v44);
    v8 = *(const __m128i **)(a2 + 40);
    LODWORD(v44) = v44 + 1;
    v22 = (unsigned int)v44;
    v23 = (unsigned int)v44;
    v24 = (unsigned int)v44 + 1LL;
    v25 = HIDWORD(v44);
  }
  else
  {
    v21 = 128;
    v25 = 128;
    v24 = 2;
    v22 = 1;
    v23 = 1;
    v20 = 3;
  }
  v26 = (unsigned int)(*(_DWORD *)(a2 + 64) - 1);
  if ( (unsigned int)v26 > v20 )
  {
    v27 = &v43;
    while ( 1 )
    {
      v28 = _mm_loadu_si128((const __m128i *)((char *)v8 + 40 * v20));
      if ( v23 + 1 > v25 )
      {
        v40.m128i_i64[0] = (__int64)v27;
        v39 = v28;
        sub_C8D5F0((__int64)v27, v45, v23 + 1, 0x10u, v24, v21);
        v23 = (unsigned int)v44;
        v28 = _mm_load_si128(&v39);
        v27 = (unsigned __int64 **)v40.m128i_i64[0];
      }
      ++v20;
      *(__m128i *)&v43[2 * v23] = v28;
      v8 = *(const __m128i **)(a2 + 40);
      v23 = (unsigned int)(v44 + 1);
      v26 = (unsigned int)(*(_DWORD *)(a2 + 64) - 1);
      LODWORD(v44) = v44 + 1;
      if ( (unsigned int)v26 <= v20 )
        break;
      v25 = HIDWORD(v44);
    }
    v22 = (unsigned int)v23;
    v21 = HIDWORD(v44);
    v24 = (unsigned int)v23 + 1LL;
  }
  v29 = *(_QWORD *)(v8->m128i_i64[5 * v26] + 96);
  v30 = *(_QWORD **)(v29 + 24);
  if ( *(_DWORD *)(v29 + 32) > 0x40u )
    v30 = (_QWORD *)*v30;
  v31 = _mm_loadu_si128(v8);
  if ( v24 > v21 )
  {
    v40 = v31;
    sub_C8D5F0((__int64)&v43, v45, v24, 0x10u, v24, v21);
    v22 = (unsigned int)v44;
    v31 = _mm_load_si128(&v40);
  }
  *(__m128i *)&v43[2 * v22] = v31;
  v32 = *(_QWORD **)(a1 + 64);
  v33 = *(unsigned int *)(a2 + 68);
  LODWORD(v44) = v44 + 1;
  switch ( (int)v10 )
  {
    case 10314:
      v34 = v30 == 0 ? 4946 : 4949;
      break;
    case 10315:
      v34 = 4947 - ((v30 == 0) - 1);
      break;
    case 10316:
      v34 = 4950 - ((v30 == 0) - 1);
      break;
    case 10317:
      v34 = 4952 - ((v30 == 0) - 1);
      break;
    case 10318:
      v34 = 4954 - ((v30 == 0) - 1);
      break;
    case 10319:
      v34 = 4956 - ((v30 == 0) - 1);
      break;
    case 10320:
      v34 = 4958 - ((v30 == 0) - 1);
      break;
    case 10321:
      v34 = v30 == 0 ? 4960 : 4963;
      break;
    case 10322:
      v34 = 4961 - ((v30 == 0) - 1);
      break;
    case 10323:
      v34 = 4964 - ((v30 == 0) - 1);
      break;
    case 10324:
      v34 = 4966 - ((v30 == 0) - 1);
      break;
    case 10325:
      v34 = 4968 - ((v30 == 0) - 1);
      break;
    case 10326:
      v34 = 4970 - ((v30 == 0) - 1);
      break;
    case 10327:
      v34 = v30 == 0 ? 4972 : 4977;
      break;
    case 10328:
      v34 = 4973 - ((v30 == 0) - 1);
      break;
    case 10329:
      v34 = 4975 - ((v30 == 0) - 1);
      break;
    case 10330:
      v34 = 4978 - ((v30 == 0) - 1);
      break;
    case 10331:
      v34 = 4980 - ((v30 == 0) - 1);
      break;
    case 10332:
      v34 = 4982 - ((v30 == 0) - 1);
      break;
    case 10333:
      v34 = 4984 - ((v30 == 0) - 1);
      break;
    case 10334:
      v34 = 4986 - ((v30 == 0) - 1);
      break;
    case 10335:
      v34 = v30 == 0 ? 4988 : 4993;
      break;
    case 10336:
      v34 = 4989 - ((v30 == 0) - 1);
      break;
    case 10337:
      v34 = 4991 - ((v30 == 0) - 1);
      break;
    case 10338:
      v34 = 4994 - ((v30 == 0) - 1);
      break;
    case 10339:
      v34 = 4996 - ((v30 == 0) - 1);
      break;
    case 10340:
      v34 = 4998 - ((v30 == 0) - 1);
      break;
    case 10341:
      v34 = 5000 - ((v30 == 0) - 1);
      break;
    case 10342:
      v34 = 5002 - ((v30 == 0) - 1);
      break;
    case 10343:
      v34 = v30 == 0 ? 5004 : 5009;
      break;
    case 10344:
      v34 = 5005 - ((v30 == 0) - 1);
      break;
    case 10345:
      v34 = 5007 - ((v30 == 0) - 1);
      break;
    case 10346:
      v34 = 5010 - ((v30 == 0) - 1);
      break;
    case 10347:
      v34 = 5012 - ((v30 == 0) - 1);
      break;
    case 10348:
      v34 = 5014 - ((v30 == 0) - 1);
      break;
    case 10349:
      v34 = 5016 - ((v30 == 0) - 1);
      break;
    case 10350:
      v34 = 5018 - ((v30 == 0) - 1);
      break;
    default:
      BUG();
  }
  v35 = sub_33E66D0(v32, v34, (__int64)&v41, *(_QWORD *)(a2 + 48), v33, v21, v43, (unsigned int)v44);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v35, v36, v37, v38);
  sub_3421DB0(v35);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v43 != (unsigned __int64 *)v45 )
    _libc_free((unsigned __int64)v43);
  if ( v41 )
    sub_B91220((__int64)&v41, v41);
}
