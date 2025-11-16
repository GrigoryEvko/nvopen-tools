// Function: sub_EA2700
// Address: 0xea2700
//
__int64 __fastcall sub_EA2700(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r14
  unsigned int v3; // r13d
  __int64 v5; // [rsp+8h] [rbp-58h] BYREF
  const char *v6; // [rsp+10h] [rbp-50h] BYREF
  char v7; // [rsp+30h] [rbp-30h]
  char v8; // [rsp+31h] [rbp-2Fh]

  v1 = sub_ECD7B0(a1);
  v2 = sub_ECD6A0(v1);
  if ( (unsigned __int8)sub_EA2660(a1, &v5) )
    return 1;
  v3 = sub_ECE000(a1);
  if ( (_BYTE)v3 )
    return 1;
  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 232) + 720LL))(
         *(_QWORD *)(a1 + 232),
         (unsigned int)v5) )
  {
    return v3;
  }
  v8 = 1;
  v6 = "function id already allocated";
  v7 = 3;
  return (unsigned int)sub_ECDA70(a1, v2, &v6, 0, 0);
}
