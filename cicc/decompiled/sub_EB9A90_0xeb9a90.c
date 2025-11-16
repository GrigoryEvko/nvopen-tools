// Function: sub_EB9A90
// Address: 0xeb9a90
//
__int64 __fastcall sub_EB9A90(__int64 a1)
{
  __int64 v1; // r14
  unsigned int v2; // r13d
  __int64 v4; // [rsp+0h] [rbp-60h] BYREF
  __int64 v5; // [rsp+8h] [rbp-58h]
  const char *v6; // [rsp+10h] [rbp-50h] BYREF
  char v7; // [rsp+30h] [rbp-30h]
  char v8; // [rsp+31h] [rbp-2Fh]

  v4 = 0;
  v5 = 0;
  v1 = sub_ECD690(a1 + 40);
  if ( (unsigned __int8)sub_EB61F0(a1, &v4) )
  {
    v8 = 1;
    v6 = "expected identifier";
    v7 = 3;
    return (unsigned int)sub_ECE0E0(a1, &v6, 0, 0);
  }
  v2 = sub_ECE000(a1);
  if ( (_BYTE)v2 )
    return v2;
  (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 232) + 1040LL))(
    *(_QWORD *)(a1 + 232),
    v1,
    v4,
    v5);
  return v2;
}
