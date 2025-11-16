// Function: sub_F02CC0
// Address: 0xf02cc0
//
__int64 __fastcall sub_F02CC0(int *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbp
  int v7; // eax
  double v8; // xmm0_8
  double v9; // xmm1_8
  _WORD *v11; // rdx
  _QWORD v12[3]; // [rsp-28h] [rbp-28h] BYREF
  unsigned int v13; // [rsp-10h] [rbp-10h]
  int v14; // [rsp-Ch] [rbp-Ch]
  __int64 v15; // [rsp-8h] [rbp-8h]

  v7 = *a1;
  if ( *a1 == -1 )
  {
    v11 = *(_WORD **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v11 <= 1u )
    {
      return sub_CB6200(a2, "?%", 2u);
    }
    else
    {
      *v11 = 9535;
      *(_QWORD *)(a2 + 32) += 2LL;
      return a2;
    }
  }
  else
  {
    v15 = v6;
    v8 = (double)v7 * 4.656612873077393e-10 * 100.0 * 100.0;
    v9 = fabs(v8);
    if ( v9 < 4.503599627370496e15 )
      *(_QWORD *)&v8 = COERCE_UNSIGNED_INT64(v9 + 4.503599627370496e15 - 4.503599627370496e15)
                     | *(_QWORD *)&v8 & 0x8000000000000000LL;
    v12[1] = "0x%08x / 0x%08x = %.2f%%";
    v13 = 0x80000000;
    v12[0] = &unk_49E4F10;
    v14 = v7;
    *(double *)&v12[2] = v8 / 100.0;
    return sub_CB6620(a2, (__int64)v12, (__int64)&unk_49E4F10, (__int64)"0x%08x / 0x%08x = %.2f%%", (__int64)a1, a6);
  }
}
