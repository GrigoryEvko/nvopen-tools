// Function: sub_30D0F50
// Address: 0x30d0f50
//
unsigned __int64 __fastcall sub_30D0F50(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  __int64 v3; // rsi

  result = 0xFFFFFFFF80000000LL;
  if ( a2 >= 0x80000000LL )
    a2 = 0x7FFFFFFF;
  if ( a2 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
    a2 = 0xFFFFFFFF80000000LL;
  v3 = *(int *)(a1 + 716) + a2;
  if ( v3 >= 0x80000000LL )
    v3 = 0x7FFFFFFF;
  if ( v3 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
    LODWORD(v3) = 0x80000000;
  *(_DWORD *)(a1 + 716) = v3;
  return result;
}
