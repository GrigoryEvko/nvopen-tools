// Function: sub_FEF7A0
// Address: 0xfef7a0
//
__int64 __fastcall sub_FEF7A0(__int64 a1, __int64 *a2)
{
  bool v2; // zf
  __int64 v3; // rax
  int v5; // edx
  __int64 v6; // [rsp+10h] [rbp-20h] BYREF
  int v7; // [rsp+18h] [rbp-18h]

  v2 = !sub_FEF380(a1, a2);
  v3 = a2[1];
  if ( v2 )
    return sub_FEF580(a1, *(_QWORD *)v3);
  v5 = *(_DWORD *)(v3 + 16);
  v6 = *(_QWORD *)(v3 + 8);
  v7 = v5;
  return sub_FEF660(a1, &v6);
}
