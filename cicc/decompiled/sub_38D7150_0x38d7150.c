// Function: sub_38D7150
// Address: 0x38d7150
//
__int64 __fastcall sub_38D7150(__int64 a1, int a2, char a3)
{
  __int64 v3; // rbp
  _DWORD *v4; // r8
  int v5; // eax
  _DWORD v7[2]; // [rsp-10h] [rbp-10h] BYREF
  __int64 v8; // [rsp-8h] [rbp-8h]

  v4 = *(_DWORD **)(a1 + 152);
  v5 = *(_DWORD *)(a1 + 124);
  if ( !a3 )
  {
    v4 = *(_DWORD **)(a1 + 144);
    v5 = *(_DWORD *)(a1 + 120);
  }
  if ( !v4 )
    return 0xFFFFFFFFLL;
  v8 = v3;
  v7[0] = a2;
  v7[1] = 0;
  return (unsigned int)sub_38D6EC0(v4, (__int64)&v4[2 * v5], v7)[1];
}
