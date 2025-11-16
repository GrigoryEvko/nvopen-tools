// Function: sub_16A57B0
// Address: 0x16a57b0
//
__int64 __fastcall sub_16A57B0(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v2; // rax
  unsigned int v3; // r8d
  unsigned __int64 v4; // rdx

  v1 = *(unsigned int *)(a1 + 8);
  LODWORD(v2) = ((unsigned __int64)(v1 + 63) >> 6) - 1;
  if ( (unsigned __int64)(v1 + 63) >> 6 )
  {
    v2 = (int)v2;
    v3 = 0;
    while ( 1 )
    {
      v4 = *(_QWORD *)(*(_QWORD *)a1 + 8 * v2);
      if ( v4 )
        break;
      --v2;
      v3 += 64;
      if ( (_DWORD)v2 == -1 )
        goto LABEL_6;
    }
    _BitScanReverse64(&v4, v4);
    v3 += v4 ^ 0x3F;
  }
  else
  {
    v3 = 0;
  }
LABEL_6:
  if ( (v1 & 0x3F) != 0 )
    v3 += (unsigned __int8)v1 | 0xFFFFFFC0;
  return v3;
}
