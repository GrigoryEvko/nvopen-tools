// Function: sub_25B5F70
// Address: 0x25b5f70
//
__int16 __fastcall sub_25B5F70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // r13
  char v5; // cl
  __int16 result; // ax
  __int64 v7; // rax

  v4 = *(__int64 **)a3;
  v5 = sub_AE5020(a2, a3);
  if ( (unsigned __int64)(1LL << v5) <= 3 )
    v5 = 2;
  if ( *(_BYTE *)(a3 + 8) == 15 && *(_DWORD *)(a3 + 12) > 1u )
  {
    v7 = sub_BCE3C0(v4, 0);
    LOBYTE(result) = sub_AE5020(a2, v7);
    HIBYTE(result) = 1;
  }
  else
  {
    LOBYTE(result) = v5;
    HIBYTE(result) = 0;
  }
  return result;
}
