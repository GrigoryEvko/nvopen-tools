// Function: sub_B19658
// Address: 0xb19658
//
__int64 __fastcall sub_B19658(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v3; // al
  __int64 result; // rax
  unsigned __int8 v6; // [rsp+1Fh] [rbp-1h]

  v6 = v3;
  sub_B19440(a1);
  result = v6;
  if ( *(_DWORD *)(a3 + 72) >= *(_DWORD *)(a2 + 72) && *(_DWORD *)(a3 + 76) <= *(_DWORD *)(a2 + 76) )
    return 1;
  return result;
}
