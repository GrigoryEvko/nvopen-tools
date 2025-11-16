// Function: sub_ADF2E0
// Address: 0xadf2e0
//
unsigned __int64 __fastcall sub_ADF2E0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v8; // r9
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v17; // rbx
  __int64 v18; // rax

  v8 = a5;
  v13 = *a1;
  if ( *(_BYTE *)(v13 + 872) )
  {
    v17 = sub_B12800(a2, a3, a4, a5);
    sub_ADE610((__int64)a1, v17, a7, a8);
    return v17 | 4;
  }
  else
  {
    v14 = a1[4];
    if ( !v14 )
    {
      v18 = sub_B6E160(v13, 71, 0, 0);
      v8 = a5;
      a1[4] = v18;
      v14 = v18;
    }
    v15 = sub_ADF0D0((__int64)a1, v14, a2, a3, a4, v8, a7, a8);
    *(_WORD *)(v15 + 2) = *(_WORD *)(v15 + 2) & 0xFFFC | 1;
    return v15 & 0xFFFFFFFFFFFFFFFBLL;
  }
}
