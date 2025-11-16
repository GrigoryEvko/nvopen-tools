// Function: sub_2DA2510
// Address: 0x2da2510
//
void __fastcall sub_2DA2510(__int64 a1)
{
  if ( &_pthread_key_create )
    _InterlockedAdd((volatile signed __int32 *)(a1 + 8), 1u);
  else
    ++*(_DWORD *)(a1 + 8);
}
