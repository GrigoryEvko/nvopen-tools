// Function: sub_22A6C90
// Address: 0x22a6c90
//
__int64 __fastcall sub_22A6C90(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v3; // r13
  char v4; // bl
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v8; // rax
  _QWORD v9[6]; // [rsp+0h] [rbp-30h] BYREF

  v2 = *(__int64 **)(*(_QWORD *)a1 + 16LL);
  v3 = *v2;
  if ( *(_BYTE *)(*v2 + 8) == 20 )
  {
    v8 = *(_QWORD *)(v3 + 24);
    if ( *(_QWORD *)(v3 + 32) == 9 && *(_QWORD *)v8 == 0x756F79614C2E7864LL && *(_BYTE *)(v8 + 8) == 116 )
      return **(unsigned int **)(v3 + 40);
  }
  v4 = sub_AE5020(a2, v3);
  v5 = sub_9208B0(a2, v3);
  v9[1] = v6;
  v9[0] = ((1LL << v4) + ((unsigned __int64)(v5 + 7) >> 3) - 1) >> v4 << v4;
  return sub_CA1930(v9);
}
