// Function: sub_3145B40
// Address: 0x3145b40
//
__int64 __fastcall sub_3145B40(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  _QWORD v8[2]; // [rsp+8h] [rbp-48h] BYREF
  __int64 v9; // [rsp+18h] [rbp-38h] BYREF
  __int64 v10; // [rsp+20h] [rbp-30h] BYREF
  unsigned __int64 v11[5]; // [rsp+28h] [rbp-28h] BYREF

  v8[0] = a2;
  *(_QWORD *)(a1 + 16) = 0;
  *(_OWORD *)a1 = 0;
  v9 = sub_A747B0(v8, -1, "statepoint-id", 0xDu);
  if ( sub_A71840((__int64)&v9) )
  {
    v3 = sub_A72240(&v9);
    if ( !sub_C93C90(v3, v4, 0xAu, v11) )
    {
      v5 = v11[0];
      *(_BYTE *)(a1 + 16) = 1;
      *(_QWORD *)(a1 + 8) = v5;
    }
  }
  v10 = sub_A747B0(v8, -1, "statepoint-num-patch-bytes", 0x1Au);
  if ( !sub_A71840((__int64)&v10) )
    return a1;
  v6 = sub_A72240(&v10);
  if ( sub_C93C90(v6, v7, 0xAu, v11) || v11[0] != LODWORD(v11[0]) )
    return a1;
  *(_DWORD *)a1 = v11[0];
  *(_BYTE *)(a1 + 4) = 1;
  return a1;
}
