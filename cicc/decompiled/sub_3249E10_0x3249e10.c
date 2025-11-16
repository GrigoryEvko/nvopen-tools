// Function: sub_3249E10
// Address: 0x3249e10
//
void __fastcall sub_3249E10(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  unsigned __int8 v4; // al
  __int64 *v5; // rcx

  v3 = a3;
  if ( *(_BYTE *)a3 != 16 )
  {
    v4 = *(_BYTE *)(a3 - 16);
    if ( (v4 & 2) != 0 )
      v5 = *(__int64 **)(a3 - 32);
    else
      v5 = (__int64 *)(a3 - 16 - 8LL * ((v4 >> 2) & 0xF));
    v3 = *v5;
  }
  sub_3249CA0(a1, a2, *(_DWORD *)(a3 + 16), v3);
}
