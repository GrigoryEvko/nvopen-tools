// Function: sub_C8B230
// Address: 0xc8b230
//
__int64 __fastcall sub_C8B230(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  sub_C8B170(a1);
  for ( result = 0; result != 20; result += 4 )
    *(_DWORD *)(a2 + result) = _byteswap_ulong(*(_DWORD *)(a1 + result + 64));
  return result;
}
