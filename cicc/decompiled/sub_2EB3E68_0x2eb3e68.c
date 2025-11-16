// Function: sub_2EB3E68
// Address: 0x2eb3e68
//
__int64 __fastcall sub_2EB3E68(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v3; // al
  __int64 result; // rax
  unsigned __int8 v6; // [rsp+1Fh] [rbp-1h]

  v6 = v3;
  sub_2EB3C30(a1);
  result = v6;
  if ( *(_DWORD *)(a3 + 72) >= *(_DWORD *)(a2 + 72) && *(_DWORD *)(a3 + 76) <= *(_DWORD *)(a2 + 76) )
    return 1;
  return result;
}
