// Function: sub_16AF620
// Address: 0x16af620
//
__int64 __fastcall sub_16AF620(int *a1, __int64 a2)
{
  __int64 v2; // rbp
  int v3; // eax
  double v4; // xmm0_8
  double v5; // xmm1_8
  _WORD *v7; // rdx
  _QWORD v8[3]; // [rsp-28h] [rbp-28h] BYREF
  unsigned int v9; // [rsp-10h] [rbp-10h]
  int v10; // [rsp-Ch] [rbp-Ch]
  __int64 v11; // [rsp-8h] [rbp-8h]

  v3 = *a1;
  if ( *a1 == -1 )
  {
    v7 = *(_WORD **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v7 <= 1u )
    {
      return sub_16E7EE0(a2, "?%", 2);
    }
    else
    {
      *v7 = 9535;
      *(_QWORD *)(a2 + 24) += 2LL;
      return a2;
    }
  }
  else
  {
    v11 = v2;
    v4 = (double)v3 * 4.656612873077393e-10 * 100.0 * 100.0;
    v5 = fabs(v4);
    if ( v5 < 4.503599627370496e15 )
      *(_QWORD *)&v4 = COERCE_UNSIGNED_INT64(v5 + 4.503599627370496e15 - 4.503599627370496e15)
                     | *(_QWORD *)&v4 & 0x8000000000000000LL;
    v8[1] = "0x%08x / 0x%08x = %.2f%%";
    v9 = 0x80000000;
    v8[0] = &unk_49EEAB0;
    v10 = v3;
    *(double *)&v8[2] = v4 / 100.0;
    return sub_16E8450(a2, v8);
  }
}
