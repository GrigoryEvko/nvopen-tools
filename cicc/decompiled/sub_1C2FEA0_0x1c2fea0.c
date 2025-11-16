// Function: sub_1C2FEA0
// Address: 0x1c2fea0
//
__int64 __fastcall sub_1C2FEA0(__int64 a1, int a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r13d
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // r14
  __int64 v8; // rax
  bool v9; // cc
  _QWORD *v10; // rax
  _QWORD v12[5]; // [rsp+8h] [rbp-28h] BYREF

  v12[0] = 0;
  v2 = sub_1C2E420(a1, "grid_constant", 0xDu, v12);
  if ( !(_BYTE)v2 )
    return 0;
  v3 = v2;
  v4 = v12[0];
  v5 = *(unsigned int *)(v12[0] + 8LL);
  if ( !(_DWORD)v5 )
    return 0;
  v6 = *(unsigned int *)(v12[0] + 8LL);
  v7 = 0;
  while ( 1 )
  {
    v8 = sub_1C2E0F0(v4 + 8 * (v7 - v5));
    v9 = *(_DWORD *)(v8 + 32) <= 0x40u;
    v10 = *(_QWORD **)(v8 + 24);
    if ( !v9 )
      v10 = (_QWORD *)*v10;
    if ( a2 == (_DWORD)v10 )
      break;
    if ( v6 == ++v7 )
      return 0;
    v4 = v12[0];
    v5 = *(unsigned int *)(v12[0] + 8LL);
  }
  return v3;
}
