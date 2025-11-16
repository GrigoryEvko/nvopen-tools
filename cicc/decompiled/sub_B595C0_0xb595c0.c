// Function: sub_B595C0
// Address: 0xb595c0
//
__int64 __fastcall sub_B595C0(__int64 a1)
{
  __int64 v1; // r8
  unsigned __int8 *v2; // rdx

  v1 = 0;
  v2 = *(unsigned __int8 **)(*(_QWORD *)(a1 + 32 * (4LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))) + 24LL);
  if ( (unsigned int)*v2 - 1 <= 1 )
    return *((_QWORD *)v2 + 17);
  return v1;
}
