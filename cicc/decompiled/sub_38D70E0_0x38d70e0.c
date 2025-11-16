// Function: sub_38D70E0
// Address: 0x38d70e0
//
__int64 __fastcall sub_38D70E0(__int64 a1, int a2, char a3)
{
  __int64 v3; // rbp
  _DWORD *v4; // r8
  int v5; // eax
  _DWORD *v6; // rax
  int v7; // r9d
  _DWORD *v8; // r10
  _DWORD v10[2]; // [rsp-10h] [rbp-10h] BYREF
  __int64 v11; // [rsp-8h] [rbp-8h]

  v4 = *(_DWORD **)(a1 + 136);
  v5 = *(_DWORD *)(a1 + 116);
  if ( !a3 )
  {
    v4 = *(_DWORD **)(a1 + 128);
    v5 = *(_DWORD *)(a1 + 112);
  }
  if ( !v4 )
    return 0xFFFFFFFFLL;
  v11 = v3;
  v10[0] = a2;
  v10[1] = 0;
  v6 = sub_38D6EC0(v4, (__int64)&v4[2 * v5], v10);
  if ( v8 == v6 || *v6 != v7 )
    return 0xFFFFFFFFLL;
  else
    return (unsigned int)v6[1];
}
