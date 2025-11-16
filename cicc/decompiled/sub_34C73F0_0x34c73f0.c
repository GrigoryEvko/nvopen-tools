// Function: sub_34C73F0
// Address: 0x34c73f0
//
__int64 __fastcall sub_34C73F0(float *a1, float *a2)
{
  float v2; // r8d
  float v3; // r9d
  unsigned int v4; // edx
  unsigned int v5; // ecx
  __int64 result; // rax
  float v7; // xmm0_4
  float v8; // xmm1_4
  bool v9; // dl

  v2 = *a1;
  v3 = *a2;
  v4 = *(_DWORD *)a1 - 1;
  v5 = *(_DWORD *)a2 - 1;
  if ( v4 <= 0x3FFFFFFE == v5 <= 0x3FFFFFFE )
  {
    v7 = a1[1];
    v8 = a2[1];
    v9 = LODWORD(v2) < LODWORD(v3);
    if ( v7 != v8 )
      v9 = v7 > v8;
    result = 0xFFFFFFFFLL;
    if ( !v9 )
    {
      LOBYTE(result) = LODWORD(v2) > LODWORD(v3);
      if ( v7 != v8 )
        LOBYTE(result) = v8 > v7;
      return (unsigned __int8)result;
    }
  }
  else
  {
    result = 0xFFFFFFFFLL;
    if ( v4 > 0x3FFFFFFE )
      return v5 <= 0x3FFFFFFE;
  }
  return result;
}
