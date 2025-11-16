// Function: sub_3897F00
// Address: 0x3897f00
//
__int64 __fastcall sub_3897F00(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  unsigned int v11; // r12d
  __int64 v13; // r13
  int v14; // eax
  double v15; // xmm4_8
  double v16; // xmm5_8
  __int64 v17; // [rsp+18h] [rbp-58h] BYREF
  const void *v18[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v19[8]; // [rsp+30h] [rbp-40h] BYREF

  v9 = *(_BYTE **)(a1 + 72);
  v10 = *(_QWORD *)(a1 + 80);
  v18[0] = v19;
  sub_3887850((__int64 *)v18, v9, (__int64)&v9[v10]);
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 3, "expected '=' here")
    || (unsigned __int8)sub_388AF10(a1, 14, "Expected '!' here")
    || (unsigned __int8)sub_388AF10(a1, 8, "Expected '{' here") )
  {
LABEL_2:
    v11 = 1;
  }
  else
  {
    v13 = sub_1632440(*(_QWORD *)(a1 + 176), v18[0], (size_t)v18[1]);
    v14 = *(_DWORD *)(a1 + 64);
    if ( v14 != 9 )
    {
      while ( 1 )
      {
        v17 = 0;
        if ( v14 != 376 || sub_2241AC0(a1 + 72, "DIExpression") )
        {
          if ( (unsigned __int8)sub_388AF10(a1, 14, "Expected '!' here")
            || (unsigned __int8)sub_3897BF0(a1, &v17, a2, a3, a4, a5, v15, v16, a8, a9) )
          {
            goto LABEL_2;
          }
        }
        else if ( (unsigned __int8)sub_388E560(a1, &v17, 0) )
        {
          goto LABEL_2;
        }
        sub_1623CA0(v13, v17);
        if ( *(_DWORD *)(a1 + 64) != 4 )
          break;
        v14 = sub_3887100(a1 + 8);
        *(_DWORD *)(a1 + 64) = v14;
      }
    }
    v11 = sub_388AF10(a1, 9, "expected end of metadata node");
  }
  if ( v18[0] != v19 )
    j_j___libc_free_0((unsigned __int64)v18[0]);
  return v11;
}
