// Function: sub_38D71A0
// Address: 0x38d71a0
//
__int64 __fastcall sub_38D71A0(__int64 a1, int a2)
{
  __int64 v2; // rbp
  _DWORD *v3; // r8
  __int64 v4; // rax
  _DWORD *v5; // rax
  int v6; // r9d
  _DWORD *v7; // r10
  _DWORD v9[2]; // [rsp-10h] [rbp-10h] BYREF
  __int64 v10; // [rsp-8h] [rbp-8h]

  v3 = *(_DWORD **)(a1 + 152);
  v4 = *(unsigned int *)(a1 + 124);
  if ( !v3 )
    return 0xFFFFFFFFLL;
  v10 = v2;
  v9[0] = a2;
  v9[1] = 0;
  v5 = sub_38D6EC0(v3, (__int64)&v3[2 * v4], v9);
  if ( v7 == v5 || *v5 != v6 )
    return 0xFFFFFFFFLL;
  else
    return (unsigned int)v5[1];
}
