// Function: sub_1BB4260
// Address: 0x1bb4260
//
__int64 __fastcall sub_1BB4260(__int64 a1, __int64 *a2, char a3, __m128i a4, __m128i a5)
{
  unsigned int v6; // eax
  unsigned int v7; // r14d
  unsigned int v8; // ecx
  int v9; // eax
  __int64 v11; // rbx
  _QWORD *v12; // r14
  __int64 v13; // rax
  char *v14; // rsi
  size_t v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // rax
  _QWORD v22[11]; // [rsp+0h] [rbp-200h] BYREF
  _BYTE v23[424]; // [rsp+58h] [rbp-1A8h] BYREF

  if ( **(_BYTE **)(*(_QWORD *)(a2[40] + 48) + 8LL) && (unsigned __int8)sub_14A2910(a2[41]) )
  {
    v16 = a2[37];
    v12 = (_QWORD *)a2[45];
    v17 = sub_1BF18B0(a2[47], *(double *)a4.m128i_i64, *(double *)a5.m128i_i64);
    sub_1BF1750(v22, v17, "CantVersionLoopWithDivergentTarget", 34, v16, 0);
    v14 = "runtime pointer checks needed. Not enabled for divergent target";
    v15 = 63;
    goto LABEL_11;
  }
  v6 = sub_1474220(*(_QWORD *)(a2[38] + 112), a2[37]);
  v7 = v6;
  if ( !a3 )
  {
    v9 = sub_1BB3F50((__int64)a2, 0, v6, a4, a5);
    *(_BYTE *)(a1 + 4) = 1;
    *(_DWORD *)a1 = v9;
    return a1;
  }
  if ( **(_BYTE **)(*(_QWORD *)(a2[40] + 48) + 8LL) )
  {
    v11 = a2[37];
    v12 = (_QWORD *)a2[45];
    v13 = sub_1BF18B0(a2[47], *(double *)a4.m128i_i64, *(double *)a5.m128i_i64);
    sub_1BF1750(v22, v13, "CantVersionLoopWithOptForSize", 29, v11, 0);
    v14 = "runtime pointer checks needed. Enable vectorization of this loop with '#pragma clang loop vectorize(enable)' w"
          "hen compiling with -Os/-Oz";
    v15 = 136;
LABEL_11:
    sub_15CAB20((__int64)v22, v14, v15);
    sub_143AA50(v12, (__int64)v22);
    v22[0] = &unk_49ECF68;
    sub_1897B80((__int64)v23);
    *(_BYTE *)(a1 + 4) = 0;
    return a1;
  }
  if ( v6 <= 1 )
  {
    v20 = a2[37];
    v12 = (_QWORD *)a2[45];
    v21 = sub_1BF18B0(a2[47], *(double *)a4.m128i_i64, *(double *)a5.m128i_i64);
    sub_1BF1750(v22, v21, "UnknownLoopCountComplexCFG", 26, v20, 0);
    v14 = "unable to calculate the loop count due to complex control flow";
    v15 = 62;
    goto LABEL_11;
  }
  v8 = sub_1BB3F50((__int64)a2, 1u, v6, a4, a5);
  if ( v7 % v8 )
  {
    v18 = a2[37];
    v12 = (_QWORD *)a2[45];
    v19 = sub_1BF18B0(a2[47], *(double *)a4.m128i_i64, *(double *)a5.m128i_i64);
    sub_1BF1750(v22, v19, "NoTailLoopWithOptForSize", 24, v18, 0);
    v14 = "cannot optimize for size and vectorize at the same time. Enable vectorization of this loop with '#pragma clang"
          " loop vectorize(enable)' when compiling with -Os/-Oz";
    v15 = 162;
    goto LABEL_11;
  }
  *(_BYTE *)(a1 + 4) = 1;
  *(_DWORD *)a1 = v8;
  return a1;
}
