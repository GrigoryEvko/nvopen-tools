// Function: sub_2726610
// Address: 0x2726610
//
__int16 __fastcall sub_2726610(__int64 a1, __int64 a2, __int64 *a3)
{
  _QWORD *v3; // rax
  __int64 v4; // rdx
  __int64 *v5; // rax
  unsigned int v6; // edx
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  char v9; // dl
  __int16 result; // ax
  __int64 v11; // rdx

  v3 = sub_DCFA50(a3, a1, a2);
  if ( *((_WORD *)v3 + 12) )
    return 0;
  v4 = v3[4];
  v5 = *(__int64 **)(v4 + 24);
  v6 = *(_DWORD *)(v4 + 32);
  if ( v6 <= 0x40 )
  {
    if ( !v6 )
    {
LABEL_11:
      v11 = *(_QWORD *)(a2 + 32);
      v8 = *(_QWORD *)(v11 + 24);
      if ( *(_DWORD *)(v11 + 32) > 0x40u )
        v8 = *(_QWORD *)v8;
      v9 = 0;
      if ( !v8 )
        goto LABEL_7;
      goto LABEL_6;
    }
    v7 = (__int64)((_QWORD)v5 << (64 - (unsigned __int8)v6)) >> (64 - (unsigned __int8)v6);
  }
  else
  {
    v7 = *v5;
  }
  if ( !v7 )
    goto LABEL_11;
  v8 = abs64(v7);
  if ( (v8 & (v8 - 1)) == 0 )
  {
LABEL_6:
    _BitScanReverse64(&v8, v8);
    v9 = 63 - (v8 ^ 0x3F);
LABEL_7:
    LOBYTE(result) = v9;
    HIBYTE(result) = 1;
    return result;
  }
  return 0;
}
