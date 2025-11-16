// Function: sub_2167470
// Address: 0x2167470
//
__int64 __fastcall sub_2167470(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned __int8 v3; // al
  unsigned int v4; // r8d

  v2 = *(_QWORD *)(a1 + 24);
  v3 = sub_2167220(*(_QWORD *)(a1 + 8), a2);
  v4 = 0;
  if ( v3 && *(_QWORD *)(v2 + 8LL * v3 + 120) )
    LOBYTE(v4) = (*(_BYTE *)(v2 + 259LL * v3 + 2586) & 0xFB) == 0;
  return v4;
}
