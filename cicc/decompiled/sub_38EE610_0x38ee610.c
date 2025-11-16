// Function: sub_38EE610
// Address: 0x38ee610
//
__int64 __fastcall sub_38EE610(__int64 a1)
{
  __int64 v1; // r13
  _DWORD *v2; // rax
  unsigned int v3; // r12d
  __int64 v5; // [rsp+0h] [rbp-50h] BYREF
  __int64 v6; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v7[2]; // [rsp+10h] [rbp-40h] BYREF
  char v8; // [rsp+20h] [rbp-30h]
  char v9; // [rsp+21h] [rbp-2Fh]

  v1 = sub_3909290(a1 + 144);
  if ( !*(_BYTE *)(a1 + 845) && (unsigned __int8)sub_38E36C0(a1) )
    return 1;
  v7[0] = 0;
  if ( sub_38EB6A0(a1, &v5, (__int64)v7) )
    return 1;
  v2 = *(_DWORD **)(a1 + 152);
  v6 = 0;
  if ( *v2 == 25 )
  {
    sub_38EB180(a1);
    if ( (unsigned __int8)sub_38EB9C0(a1, &v6) )
      return 1;
  }
  v9 = 1;
  v8 = 3;
  v7[0] = "unexpected token in '.zero' directive";
  v3 = sub_3909E20(a1, 9, v7);
  if ( (_BYTE)v3 )
    return 1;
  (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 328) + 496LL))(
    *(_QWORD *)(a1 + 328),
    v5,
    v6,
    v1);
  return v3;
}
