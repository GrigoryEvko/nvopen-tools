// Function: sub_B2BC30
// Address: 0xb2bc30
//
__int64 __fastcall sub_B2BC30(__int64 a1, __int64 a2)
{
  int v3; // esi
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 result; // rax
  char v7; // bl
  __int64 v8; // rax
  __int64 v9; // rdx
  _QWORD v10[6]; // [rsp+0h] [rbp-30h] BYREF

  v3 = *(_DWORD *)(a1 + 32);
  v10[0] = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 120LL);
  v4 = sub_A744E0(v10, v3);
  v5 = sub_B2B600(v4);
  result = 0;
  if ( v5 )
  {
    v7 = sub_AE5020(a2, v5);
    v8 = sub_9208B0(a2, v5);
    v10[1] = v9;
    v10[0] = ((1LL << v7) + ((unsigned __int64)(v8 + 7) >> 3) - 1) >> v7 << v7;
    return sub_CA1930(v10);
  }
  return result;
}
