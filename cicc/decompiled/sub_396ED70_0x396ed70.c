// Function: sub_396ED70
// Address: 0x396ed70
//
__int64 __fastcall sub_396ED70(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // rdx
  __m128i v17; // xmm1
  __int64 v18; // rax
  unsigned __int64 v19; // r12
  __int64 v20; // r15
  __int64 *v21; // r13
  unsigned int v22; // eax
  __int64 v23; // rbx
  int v24; // edx
  __int64 v25; // rsi
  __int64 v26; // r8
  __m128i v27; // [rsp-58h] [rbp-58h] BYREF
  __m128i v28; // [rsp-48h] [rbp-48h] BYREF

  result = *(_QWORD *)(a2 + 8);
  if ( (*(_BYTE *)(result + 809) & 2) != 0 )
  {
    v4 = sub_396DD80((__int64)a1);
    v5 = sub_396E9A0((__int64)a1);
    result = sub_38D3840(v4 + 8, v5);
    v8 = result;
    if ( result )
    {
      v9 = *(_QWORD *)(a2 + 56);
      if ( !*(_BYTE *)(v9 + 36) )
      {
        v10 = a1[32];
        v11 = *(unsigned int *)(v10 + 120);
        if ( (_DWORD)v11 )
        {
          v12 = (__int64 *)(*(_QWORD *)(v10 + 112) + 32LL * (unsigned int)v11 - 32);
          v13 = v12[2];
          v14 = v12[3];
          v15 = *v12;
          v16 = v12[1];
        }
        else
        {
          v16 = 0;
          v15 = 0;
          v14 = 0;
          v13 = 0;
        }
        v27.m128i_i64[0] = v15;
        v27.m128i_i64[1] = v16;
        v28.m128i_i64[0] = v13;
        v28.m128i_i64[1] = v14;
        if ( *(_DWORD *)(v10 + 124) <= (unsigned int)v11 )
        {
          sub_16CD150(v10 + 112, (const void *)(v10 + 128), 0, 32, v6, v7);
          v11 = *(unsigned int *)(v10 + 120);
        }
        v17 = _mm_loadu_si128(&v28);
        v18 = *(_QWORD *)(v10 + 112) + 32 * v11;
        *(__m128i *)v18 = _mm_loadu_si128(&v27);
        *(__m128i *)(v18 + 16) = v17;
        ++*(_DWORD *)(v10 + 120);
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD, __int64, __int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[32] + 160LL))(
          a1[32],
          v8,
          0,
          v14,
          v6,
          v7,
          v27.m128i_i64[0],
          v27.m128i_i64[1],
          v28.m128i_i64[0],
          v28.m128i_i64[1]);
        v19 = *(_QWORD *)(v9 + 48);
        v20 = a1[48];
        v21 = (__int64 *)a1[32];
        v22 = sub_15A9520(a1[29] + 16LL, *(_DWORD *)(a1[29] + 28LL));
        sub_38DDC80(v21, v20, v22, 0);
        sub_38DCDD0(a1[32], v19);
        v23 = a1[32];
        result = *(unsigned int *)(v23 + 120);
        v24 = result;
        if ( (unsigned int)result > 1 )
        {
          result = *(_QWORD *)(v23 + 112) + 32 * result;
          v25 = *(_QWORD *)(result - 64);
          v26 = *(_QWORD *)(result - 56);
          if ( *(_QWORD *)(result - 24) != v26 || *(_QWORD *)(result - 32) != v25 )
          {
            result = (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v23 + 152LL))(v23, v25, v26);
            v24 = *(_DWORD *)(v23 + 120);
          }
          *(_DWORD *)(v23 + 120) = v24 - 1;
        }
      }
    }
  }
  return result;
}
