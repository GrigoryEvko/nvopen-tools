// Function: sub_31EFD10
// Address: 0x31efd10
//
void __fastcall sub_31EFD10(__int64 a1, const void *a2, __int64 *a3, char a4)
{
  unsigned __int8 v6; // al
  const __m128i *v7; // rdi
  unsigned int *v8; // r14
  unsigned int *v9; // rbx
  __int64 (__fastcall *v10)(__int64); // rax
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 (__fastcall *v16)(__int64, __int64, __int64); // rax
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 *v19; // r15
  __int64 v20; // rax
  const __m128i *v21; // rax
  const __m128i *v22; // rdx
  __m128i v23; // xmm0
  __int32 v24; // edi
  __int64 v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // [rsp+10h] [rbp-110h]
  unsigned __int8 v28; // [rsp+1Eh] [rbp-102h]
  char v29; // [rsp+1Fh] [rbp-101h]
  const __m128i *v30; // [rsp+20h] [rbp-100h] BYREF
  __int64 v31; // [rsp+28h] [rbp-F8h]
  _BYTE v32[240]; // [rsp+30h] [rbp-F0h] BYREF

  v29 = a4;
  v30 = (const __m128i *)v32;
  v31 = 0x800000000LL;
  sub_31DFEC0(a1, a2, a3, (__int64)&v30);
  if ( (_DWORD)v31 )
  {
    if ( (*(_BYTE *)(*(_QWORD *)(a1 + 200) + 876LL) & 1) == 0 )
    {
      v21 = v30;
      v22 = (const __m128i *)((char *)v30 + 24 * (unsigned int)v31 - 24);
      if ( v30 < v22 )
      {
        do
        {
          v23 = _mm_loadu_si128(v22);
          v24 = v21->m128i_i32[0];
          v22 = (const __m128i *)((char *)v22 - 24);
          v21 = (const __m128i *)((char *)v21 + 24);
          v25 = v21[-1].m128i_i64[0];
          v26 = v21[-1].m128i_i64[1];
          *(__m128i *)((char *)v21 - 24) = v23;
          v21[-1].m128i_i64[1] = v22[2].m128i_i64[1];
          v22[1].m128i_i32[2] = v24;
          v22[2].m128i_i64[0] = v25;
          v22[2].m128i_i64[1] = v26;
        }
        while ( v21 < v22 );
      }
    }
    v6 = sub_AE4370((__int64)a2, 0);
    v7 = v30;
    v28 = v6;
    v8 = (unsigned int *)v30 + 6 * (unsigned int)v31;
    if ( v8 != (unsigned int *)v30 )
    {
      v9 = (unsigned int *)v30;
      while ( 1 )
      {
        v17 = sub_31DA6B0(a1);
        v18 = *((_QWORD *)v9 + 2);
        v19 = (__int64 *)v17;
        if ( !v18 )
          goto LABEL_17;
        if ( (*(_BYTE *)(v18 + 32) & 0xF) != 1 )
        {
          v27 = *((_QWORD *)v9 + 2);
          if ( !sub_B2FC80(v27) )
            break;
        }
LABEL_12:
        v9 += 6;
        if ( v8 == v9 )
          goto LABEL_20;
      }
      v18 = sub_31DB510(a1, v27);
LABEL_17:
      v20 = *v19;
      if ( v29 )
      {
        v10 = *(__int64 (__fastcall **)(__int64))(v20 + 152);
        if ( v10 == sub_302E420 )
        {
          v11 = v19[120];
          goto LABEL_7;
        }
      }
      else
      {
        v10 = *(__int64 (__fastcall **)(__int64))(v20 + 160);
        if ( v10 == sub_302E430 )
        {
          v11 = v19[121];
          goto LABEL_7;
        }
      }
      v11 = ((__int64 (__fastcall *)(__int64 *, _QWORD, __int64))v10)(v19, *v9, v18);
LABEL_7:
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 176LL))(*(_QWORD *)(a1 + 224), v11, 0);
      v12 = *(_QWORD *)(a1 + 224);
      v13 = *(unsigned int *)(v12 + 128);
      if ( (_DWORD)v13 )
      {
        v14 = *(_QWORD *)(v12 + 120) + 32 * v13 - 32;
        if ( *(_QWORD *)(v14 + 16) != *(_QWORD *)v14 || *(_DWORD *)(v14 + 24) != *(_DWORD *)(v14 + 8) )
          sub_31DCA70(a1, v28, 0, 0);
      }
      v15 = *((_QWORD *)v9 + 1);
      v16 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1 + 352LL);
      if ( v16 == sub_30200D0 )
        sub_31EA6F0(a1, (__int64)a2, v15, 0);
      else
        v16(a1, (__int64)a2, v15);
      goto LABEL_12;
    }
  }
  else
  {
LABEL_20:
    v7 = v30;
  }
  if ( v7 != (const __m128i *)v32 )
    _libc_free((unsigned __int64)v7);
}
