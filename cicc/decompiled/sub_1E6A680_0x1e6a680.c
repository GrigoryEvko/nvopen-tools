// Function: sub_1E6A680
// Address: 0x1e6a680
//
__int64 __fastcall sub_1E6A680(_QWORD *a1, unsigned int a2)
{
  __int64 (*v2)(); // rax
  _QWORD *v3; // rax
  unsigned __int16 *v4; // rdx
  unsigned __int16 v5; // cx
  unsigned __int16 v6; // r9
  __int64 v7; // r11
  __int64 v8; // r10
  __int16 *v9; // rsi
  __int16 *v10; // rax
  __int16 v11; // dx

  v2 = *(__int64 (**)())(**(_QWORD **)(*a1 + 16LL) + 112LL);
  if ( v2 == sub_1D00B10 || (v3 = (_QWORD *)v2()) == 0 )
    BUG();
  v4 = (unsigned __int16 *)(v3[6] + 4LL * a2);
  v5 = *v4;
  v6 = v4[1];
  if ( !*v4 )
    return 0;
  v7 = v3[7];
  v8 = v3[1];
  while ( 1 )
  {
    v9 = (__int16 *)(v7 + 2LL * *(unsigned int *)(v8 + 24LL * v5 + 8));
LABEL_6:
    v10 = v9;
    if ( !v9 )
      return 1;
    while ( (*(_QWORD *)(a1[38] + 8 * ((unsigned __int64)v5 >> 6)) & (1LL << v5)) != 0 )
    {
      v11 = *v10;
      v9 = 0;
      ++v10;
      if ( !v11 )
        goto LABEL_6;
      v5 += v11;
      if ( !v10 )
        return 1;
    }
    v5 = v6;
    if ( !v6 )
      return 0;
    v6 = 0;
  }
}
