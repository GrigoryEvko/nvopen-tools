// Function: sub_38EBE90
// Address: 0x38ebe90
//
__int64 __fastcall sub_38EBE90(__int64 a1)
{
  __int64 v1; // r12
  unsigned __int64 v3; // [rsp+18h] [rbp-58h] BYREF
  const char *v4; // [rsp+20h] [rbp-50h] BYREF
  char v5; // [rsp+30h] [rbp-40h]
  char v6; // [rsp+31h] [rbp-3Fh]
  const char *v7; // [rsp+40h] [rbp-30h] BYREF
  char v8; // [rsp+50h] [rbp-20h]
  char v9; // [rsp+51h] [rbp-1Fh]

  v1 = sub_3909290(a1 + 144);
  if ( !*(_BYTE *)(a1 + 845) && (unsigned __int8)sub_38E36C0(a1) )
    return 1;
  if ( (unsigned __int8)sub_38EB9C0(a1, &v3) )
    return 1;
  v6 = 1;
  v5 = 3;
  v4 = "unexpected token after expression in '.bundle_align_mode' directive";
  if ( (unsigned __int8)sub_3909E20(a1, 9, &v4) )
    return 1;
  v9 = 1;
  v7 = "invalid bundle alignment size (expected between 0 and 30)";
  v8 = 3;
  if ( (unsigned __int8)sub_3909C80(a1, v3 > 0x1E, v1, &v7) )
    return 1;
  (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 328) + 1024LL))(*(_QWORD *)(a1 + 328), (unsigned int)v3);
  return 0;
}
