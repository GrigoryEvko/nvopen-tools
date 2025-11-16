// Function: sub_39B0A10
// Address: 0x39b0a10
//
void __fastcall sub_39B0A10(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdi
  int v9; // r13d
  int v10; // r8d
  int v11; // r9d
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rcx
  __int64 v17; // rsi
  __m128i v18; // xmm1
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rdx
  __int64 v28; // rcx
  signed __int64 v29; // r8
  char *v30; // r9
  __int64 v31; // rbx
  __int64 v32; // rax
  int v33; // edx
  __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v36; // r8
  __int64 *v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rax
  __m128i v40; // [rsp+0h] [rbp-40h] BYREF
  __m128i v41; // [rsp+10h] [rbp-30h] BYREF

  if ( *(_BYTE *)(a1 + 24) || *(_BYTE *)(a1 + 26) )
  {
    v8 = *a2;
    v9 = 0;
    if ( (*(_BYTE *)(*a2 + 18) & 8) == 0 )
      goto LABEL_4;
  }
  else
  {
    if ( !*(_BYTE *)(a1 + 25) )
      return;
    v8 = *a2;
    v9 = 0;
    if ( (*(_BYTE *)(*a2 + 18) & 8) == 0 )
      goto LABEL_4;
  }
  v38 = sub_15E38F0(v8);
  v39 = sub_1649C60(v38);
  v9 = sub_14DD7D0(v39);
  if ( (unsigned int)(v9 - 7) <= 3 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 80LL))(a1);
    if ( v9 == 8 && *((_BYTE *)a2 + 523) )
      return;
    goto LABEL_5;
  }
LABEL_4:
  sub_1E0EC90((unsigned __int64)a2, 0, a3, a4, a5, a6);
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 80LL))(a1);
LABEL_5:
  if ( *(_BYTE *)(a1 + 24) || *(_BYTE *)(a1 + 25) )
  {
    v12 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL);
    v13 = *(unsigned int *)(v12 + 120);
    if ( (_DWORD)v13 )
    {
      v37 = (__int64 *)(*(_QWORD *)(v12 + 112) + 32LL * (unsigned int)v13 - 32);
      v17 = v37[2];
      v16 = v37[3];
      v15 = *v37;
      v14 = v37[1];
    }
    else
    {
      v14 = 0;
      v15 = 0;
      v16 = 0;
      v17 = 0;
    }
    v40.m128i_i64[0] = v15;
    v40.m128i_i64[1] = v14;
    v41.m128i_i64[0] = v17;
    v41.m128i_i64[1] = v16;
    if ( *(_DWORD *)(v12 + 124) <= (unsigned int)v13 )
    {
      sub_16CD150(v12 + 112, (const void *)(v12 + 128), 0, 32, v10, v11);
      v13 = *(unsigned int *)(v12 + 120);
    }
    v18 = _mm_loadu_si128(&v41);
    v19 = 0;
    v20 = *(_QWORD *)(v12 + 112) + 32 * v13;
    *(__m128i *)v20 = _mm_loadu_si128(&v40);
    *(__m128i *)(v20 + 16) = v18;
    ++*(_DWORD *)(v12 + 120);
    v21 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL);
    v22 = *(unsigned int *)(v21 + 120);
    if ( (_DWORD)v22 )
      v19 = *(_QWORD *)(*(_QWORD *)(v21 + 112) + 32 * v22 - 32);
    v23 = sub_38DD570(v21, v19);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD, __int64, __int64, __int64, __int64, __int64, __int64, __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 160LL))(
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
      v23,
      0,
      v24,
      v25,
      v26,
      v40.m128i_i64[0],
      v40.m128i_i64[1],
      v41.m128i_i64[0],
      v41.m128i_i64[1]);
    switch ( v9 )
    {
      case 8:
        sub_39AD190(a1, a2);
        break;
      case 7:
        sub_39AF220(a1, a2);
        break;
      case 9:
        sub_39ADBC0(a1, (__int64)a2);
        break;
      case 10:
        sub_39AF860(a1, a2);
        break;
      default:
        sub_39AB100((__int64 *)a1, v23, v27, v28, v29, v30);
        break;
    }
    v31 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL);
    v32 = *(unsigned int *)(v31 + 120);
    v33 = v32;
    if ( (unsigned int)v32 > 1 )
    {
      v34 = *(_QWORD *)(v31 + 112) + 32 * v32;
      v35 = *(_QWORD *)(v34 - 64);
      v36 = *(_QWORD *)(v34 - 56);
      if ( *(_QWORD *)(v34 - 24) != v36 || *(_QWORD *)(v34 - 32) != v35 )
      {
        (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v31 + 152LL))(v31, v35, v36);
        v33 = *(_DWORD *)(v31 + 120);
      }
      *(_DWORD *)(v31 + 120) = v33 - 1;
    }
  }
}
