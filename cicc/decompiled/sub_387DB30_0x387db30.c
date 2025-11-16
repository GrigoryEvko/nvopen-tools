// Function: sub_387DB30
// Address: 0x387db30
//
__int64 __fastcall sub_387DB30(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v13; // r15
  double v14; // xmm4_8
  double v15; // xmm5_8
  int v16; // eax
  __int64 *v17; // rax
  __int64 v19; // r13
  double v20; // xmm4_8
  double v21; // xmm5_8
  __int64 v22; // rsi
  _QWORD *v23; // rax
  __int64 v24; // rcx
  __int64 v25; // r15
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 *v29; // rbx
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // [rsp+0h] [rbp-70h] BYREF
  __int16 v33; // [rsp+10h] [rbp-60h]
  char v34[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v35; // [rsp+30h] [rbp-40h]

  v13 = sub_1452540(a2);
  v16 = *(_DWORD *)(a2 + 48);
  if ( (v16 & 1) != 0 )
  {
    v19 = (__int64)sub_387A650(a1, v13, a3, 0, a4, a5, a6, a7, v14, v15, a10, a11);
    if ( (*(_BYTE *)(a2 + 48) & 2) != 0 )
    {
      v22 = v13;
      v23 = sub_387A650(a1, v13, a3, 1, a4, a5, a6, a7, v20, v21, a10, a11);
      v25 = (__int64)v23;
      if ( v19 )
      {
        if ( !v23 )
          return v19;
        v33 = 257;
        if ( *((_BYTE *)v23 + 16) <= 0x10u )
        {
          if ( sub_1593BB0((__int64)v23, v22, 257, v24) )
            return v19;
          if ( *(_BYTE *)(v19 + 16) <= 0x10u )
          {
            v19 = sub_15A2D10((__int64 *)v19, v25, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
            v26 = sub_14DBA30(v19, a1[41], 0);
            if ( v26 )
              return v26;
            return v19;
          }
        }
        v35 = 257;
        v27 = sub_15FB440(27, (__int64 *)v19, v25, (__int64)v34, 0);
        v28 = a1[34];
        v19 = v27;
        if ( v28 )
        {
          v29 = (__int64 *)a1[35];
          sub_157E9D0(v28 + 40, v27);
          v30 = *(_QWORD *)(v19 + 24);
          v31 = *v29;
          *(_QWORD *)(v19 + 32) = v29;
          v31 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v19 + 24) = v31 | v30 & 7;
          *(_QWORD *)(v31 + 8) = v19 + 24;
          *v29 = *v29 & 7 | (v19 + 24);
        }
        sub_164B780(v19, &v32);
        sub_12A86E0(a1 + 33, v19);
        return v19;
      }
      v19 = (__int64)v23;
    }
  }
  else
  {
    if ( (v16 & 2) == 0 )
    {
LABEL_3:
      v17 = (__int64 *)sub_16498A0(a3);
      return sub_159C540(v17);
    }
    v19 = (__int64)sub_387A650(a1, v13, a3, 1, a4, a5, a6, a7, v14, v15, a10, a11);
  }
  if ( !v19 )
    goto LABEL_3;
  return v19;
}
