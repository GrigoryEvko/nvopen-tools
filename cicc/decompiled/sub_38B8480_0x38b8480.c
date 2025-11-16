// Function: sub_38B8480
// Address: 0x38b8480
//
__int64 __fastcall sub_38B8480(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r13
  int v12; // eax
  __int64 v13; // rdi
  double v14; // xmm4_8
  double v15; // xmm5_8
  unsigned __int64 v16; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  double v21; // xmm4_8
  double v22; // xmm5_8
  const char *v23; // [rsp+0h] [rbp-30h] BYREF
  char v24; // [rsp+10h] [rbp-20h]
  char v25; // [rsp+11h] [rbp-1Fh]

  v10 = a1 + 8;
  v12 = sub_3887100(a1 + 8);
  v13 = *(_QWORD *)a1;
  *(_DWORD *)(a1 + 64) = v12;
  if ( (unsigned __int8)sub_16033A0(v13) )
  {
    v16 = *(_QWORD *)(a1 + 56);
    v25 = 1;
    v24 = 3;
    v23 = "Can't read textual IR with a Context that discards named Values";
    return sub_38814C0(v10, v16, (__int64)&v23);
  }
  else if ( (unsigned __int8)sub_38B8180(a1, a3, a4, a5, a6, v14, v15, a9, a10)
         || (unsigned __int8)sub_38967E0(
                               (__int64 *)a1,
                               a2,
                               a3,
                               *(double *)a4.m128i_i64,
                               a5,
                               a6,
                               v21,
                               v22,
                               a9,
                               a10,
                               v18,
                               v19,
                               v20) )
  {
    return 1;
  }
  else
  {
    return sub_388AC50((_QWORD *)a1);
  }
}
