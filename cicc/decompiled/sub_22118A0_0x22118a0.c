// Function: sub_22118A0
// Address: 0x22118a0
//
__int64 __fastcall sub_22118A0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  char v5; // [rsp+Fh] [rbp-49h] BYREF
  _QWORD v6[4]; // [rsp+10h] [rbp-48h] BYREF
  void (__fastcall *v7)(_QWORD *); // [rsp+30h] [rbp-28h]

  v3 = *(_QWORD *)(a2 + 24);
  v7 = 0;
  sub_2221730(v3, v6);
  if ( !v7 )
    sub_426248((__int64)"uninitialized __any_string");
  sub_2216C30(a1, v6[0], v6[1], &v5);
  if ( v7 )
    v7(v6);
  return a1;
}
