// Function: sub_34CEDB0
// Address: 0x34cedb0
//
__int64 __fastcall sub_34CEDB0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rbx
  unsigned __int16 v3; // ax
  unsigned int v4; // r8d

  v2 = *(_QWORD *)(a1 + 32);
  v3 = sub_2D5BAE0(v2, *(_QWORD *)(a1 + 16), a2, 0);
  v4 = 0;
  if ( v3 && *(_QWORD *)(v2 + 8LL * v3 + 112) )
    LOBYTE(v4) = (*(_BYTE *)(v2 + 500LL * v3 + 6660) & 0xFB) == 0;
  return v4;
}
