// Function: sub_14A3770
// Address: 0x14a3770
//
char __fastcall sub_14A3770(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 (*v3)(void); // rax
  __int64 v4; // r13
  __int64 v5; // rbx
  __int64 v6; // r8
  char result; // al
  __int64 v8; // rbx

  v3 = *(__int64 (**)(void))(**(_QWORD **)a1 + 776LL);
  if ( (char *)v3 != (char *)sub_14A0E50 )
    return v3();
  v4 = a3 + 112;
  v5 = sub_1560340(a3 + 112, 0xFFFFFFFFLL, "target-cpu", 10);
  v6 = sub_1560340(a2 + 112, 0xFFFFFFFFLL, "target-cpu", 10);
  result = 0;
  if ( v5 == v6 )
  {
    v8 = sub_1560340(v4, 0xFFFFFFFFLL, "target-features", 15);
    return v8 == sub_1560340(a2 + 112, 0xFFFFFFFFLL, "target-features", 15);
  }
  return result;
}
