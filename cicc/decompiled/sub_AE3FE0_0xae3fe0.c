// Function: sub_AE3FE0
// Address: 0xae3fe0
//
__int64 __fastcall sub_AE3FE0(__int64 a1, int a2)
{
  _DWORD *v2; // r8
  __int64 v3; // rax
  _DWORD *v4; // rax
  char v5; // r9
  _DWORD *v6; // r10
  int v8; // [rsp+Ch] [rbp-4h] BYREF

  v2 = *(_DWORD **)(a1 + 64);
  v3 = *(unsigned int *)(a1 + 72);
  v8 = a2;
  v4 = sub_AE1180(v2, (__int64)&v2[2 * v3], &v8);
  if ( v6 == v4 )
    v4 -= 2;
  if ( v5 )
    return *((unsigned __int8 *)v4 + 4);
  else
    return *((unsigned __int8 *)v4 + 5);
}
