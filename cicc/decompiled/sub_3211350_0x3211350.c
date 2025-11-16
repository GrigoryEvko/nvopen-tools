// Function: sub_3211350
// Address: 0x3211350
//
__int64 __fastcall sub_3211350(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned int v3; // eax
  __int64 result; // rax
  __int64 v5; // [rsp+0h] [rbp-30h] BYREF
  int v6; // [rsp+8h] [rbp-28h]
  __int64 v7; // [rsp+10h] [rbp-20h] BYREF
  unsigned int v8; // [rsp+18h] [rbp-18h]

  v2 = sub_BA8DC0(a2, (__int64)"llvm.dbg.cu", 11);
  v3 = 0;
  if ( v2 )
    v3 = sub_B91A00(v2);
  v8 = v3;
  v7 = v2;
  sub_BA95A0((__int64)&v7);
  v5 = v2;
  v6 = 0;
  sub_BA95A0((__int64)&v5);
  result = v8;
  if ( v6 == v8 )
    *(_QWORD *)(a1 + 8) = 0;
  return result;
}
