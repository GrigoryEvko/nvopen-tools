// Function: sub_258EEC0
// Address: 0x258eec0
//
__int64 __fastcall sub_258EEC0(__int64 *a1, unsigned __int64 a2)
{
  __int64 v2; // r12
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  unsigned int v6; // r8d

  v2 = *a1;
  v3 = sub_250D2C0(a2, 0);
  v5 = sub_258DCE0(v2, v3, v4, a1[1], 1, 0, 1);
  v6 = 0;
  if ( v5 )
    LOBYTE(v6) = *(_DWORD *)(v5 + 108) != 0;
  return v6;
}
