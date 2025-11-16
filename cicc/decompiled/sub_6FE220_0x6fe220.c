// Function: sub_6FE220
// Address: 0x6fe220
//
__int64 __fastcall sub_6FE220(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  unsigned int v13; // eax
  unsigned int v14; // [rsp-2Ch] [rbp-2Ch]
  _QWORD v15[4]; // [rsp-20h] [rbp-20h] BYREF

  if ( *(_BYTE *)(a1 + 17) != 1 )
    return 0;
  v15[3] = v2;
  if ( sub_6ED0A0(a1) || (*(_BYTE *)(a1 + 20) & 1) != 0 )
    return 0;
  v15[0] = sub_724DC0(a1, a2, v4, v5, v6, v7);
  v12 = sub_6F6EB0(a1, 1, v8, v9, v10, v11);
  v13 = sub_717510(v12, v15[0], 0);
  if ( v13 )
    v13 = sub_710600(v15[0]) != 0;
  v14 = v13;
  sub_724E30(v15);
  return v14;
}
