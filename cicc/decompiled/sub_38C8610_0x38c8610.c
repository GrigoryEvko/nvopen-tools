// Function: sub_38C8610
// Address: 0x38c8610
//
void __fastcall sub_38C8610(_BYTE *a1, __int64 a2, __int64 *a3, unsigned __int64 *a4)
{
  __int64 v6; // rcx
  __int64 v8; // rax
  __int64 v9; // r14
  int v10; // eax
  int v11; // edx
  int v12; // r9d
  __int64 v13; // r8
  unsigned int v14; // eax
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // rcx
  int v19; // eax
  unsigned int v20; // r13d
  __int64 v21; // r12
  unsigned __int64 *v22; // rdi
  __m128i *v23; // rsi
  __int64 v24; // [rsp-68h] [rbp-68h]
  unsigned int v25; // [rsp-5Ch] [rbp-5Ch]
  __m128i v26; // [rsp-58h] [rbp-58h] BYREF
  __m128i v27; // [rsp-48h] [rbp-48h] BYREF

  if ( (a1[8] & 1) == 0 )
  {
    v6 = 0;
    v8 = *(unsigned int *)(a2 + 120);
    v9 = *(_QWORD *)(a2 + 8);
    if ( (_DWORD)v8 )
      v6 = *(_QWORD *)(*(_QWORD *)(a2 + 112) + 32 * v8 - 32);
    v10 = *(_DWORD *)(v9 + 1072);
    if ( v10 )
    {
      v11 = v10 - 1;
      v12 = 1;
      v13 = *(_QWORD *)(v9 + 1056);
      v14 = (v10 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v15 = *(_QWORD *)(v13 + 8LL * v14);
      if ( v15 == v6 )
      {
LABEL_6:
        if ( (*a1 & 4) != 0 )
        {
          v16 = *((_QWORD *)a1 - 1);
          v17 = *(_QWORD *)v16;
          v18 = v16 + 16;
          if ( *(_QWORD *)v16 && *(_BYTE *)(v16 + 16) == 95 )
          {
            --v17;
            v18 = v16 + 17;
          }
        }
        else
        {
          v17 = 0;
          v18 = 0;
        }
        v24 = v18;
        v25 = *(_DWORD *)(v9 + 1044);
        v19 = sub_16CE270(a3, *a4);
        v20 = sub_16CFA40(a3, *a4, v19);
        v21 = sub_38BFA60(v9, 1);
        (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 176LL))(a2, v21, 0);
        v22 = *(unsigned __int64 **)(a2 + 8);
        v26.m128i_i64[1] = v17;
        v26.m128i_i64[0] = v24;
        v23 = (__m128i *)v22[139];
        v27.m128i_i64[0] = __PAIR64__(v20, v25);
        v27.m128i_i64[1] = v21;
        if ( v23 == (__m128i *)v22[140] )
        {
          sub_38C8480(v22 + 138, v23, &v26);
        }
        else
        {
          if ( v23 )
          {
            *v23 = _mm_loadu_si128(&v26);
            v23[1] = _mm_loadu_si128(&v27);
            v23 = (__m128i *)v22[139];
          }
          v22[139] = (unsigned __int64)&v23[2];
        }
      }
      else
      {
        while ( v15 != -8 )
        {
          v14 = v11 & (v12 + v14);
          v15 = *(_QWORD *)(v13 + 8LL * v14);
          if ( v15 == v6 )
            goto LABEL_6;
          ++v12;
        }
      }
    }
  }
}
