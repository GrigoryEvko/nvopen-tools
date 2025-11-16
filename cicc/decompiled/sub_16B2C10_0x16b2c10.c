// Function: sub_16B2C10
// Address: 0x16b2c10
//
__int64 __fastcall sub_16B2C10(void (__fastcall ***a1)(_QWORD), __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rdx

  v2 = *(_QWORD *)(a2 + 32);
  (**a1)(a1);
  if ( v3 )
  {
    if ( *(_QWORD *)(a2 + 64) )
      v3 = *(_QWORD *)(a2 + 64);
    v2 += v3 + (-(__int64)(((*(_BYTE *)(a2 + 13) >> 1) & 2) == 0) & 0xFFFFFFFFFFFFFFFDLL) + 6;
  }
  return v2 + 6;
}
