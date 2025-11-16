// Function: sub_EAFDE0
// Address: 0xeafde0
//
__int64 __fastcall sub_EAFDE0(__int64 a1)
{
  __int64 v1; // r13
  _DWORD *v2; // rax
  unsigned int v3; // r12d
  __int64 v5; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v1 = sub_ECD690(a1 + 40);
  if ( !*(_BYTE *)(a1 + 869) && (unsigned __int8)sub_EA2540(a1) )
    return 1;
  v6[0] = 0;
  if ( sub_EAC4D0(a1, &v5, (__int64)v6) )
    return 1;
  v2 = *(_DWORD **)(a1 + 48);
  v6[0] = 0;
  if ( *v2 == 26 )
  {
    sub_EABFE0(a1);
    if ( (unsigned __int8)sub_EAC8B0(a1, v6) )
      return 1;
  }
  v3 = sub_ECE000(a1);
  if ( (_BYTE)v3 )
    return 1;
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD, __int64))(**(_QWORD **)(a1 + 232) + 584LL))(
    *(_QWORD *)(a1 + 232),
    v5,
    v6[0],
    v1);
  return v3;
}
