// Function: sub_388BF60
// Address: 0x388bf60
//
__int64 __fastcall sub_388BF60(__int64 a1, _DWORD *a2)
{
  *a2 = 0;
  if ( *(_DWORD *)(a1 + 64) != 89 )
    return 0;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' in address space") || (unsigned __int8)sub_388BA90(a1, a2) )
    return 1;
  else
    return sub_388AF10(a1, 13, "expected ')' in address space");
}
