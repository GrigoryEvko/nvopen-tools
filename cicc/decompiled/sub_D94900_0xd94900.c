// Function: sub_D94900
// Address: 0xd94900
//
void __fastcall sub_D94900(__int64 a1, unsigned int a2)
{
  unsigned int v2; // eax
  __int64 v3; // rdx
  __int64 v4; // r8
  __int64 v5; // rdx
  bool v6; // zf
  __int64 v7; // rsi
  unsigned __int64 v8; // rdx

  v2 = *(_DWORD *)(a1 + 8);
  if ( v2 > 0x40 )
  {
    sub_C44B70(a1, a2);
  }
  else
  {
    v3 = 0;
    if ( v2 )
      v3 = (__int64)(*(_QWORD *)a1 << (64 - (unsigned __int8)v2)) >> (64 - (unsigned __int8)v2);
    v4 = v3 >> 63;
    v5 = v3 >> a2;
    v6 = a2 == v2;
    v7 = v4;
    if ( !v6 )
      v7 = v5;
    v8 = v7 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v2);
    if ( !v2 )
      v8 = 0;
    *(_QWORD *)a1 = v8;
  }
}
