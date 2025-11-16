// Function: sub_EB97E0
// Address: 0xeb97e0
//
__int64 __fastcall sub_EB97E0(__int64 a1)
{
  bool v1; // zf
  __int64 v2; // rsi
  __int64 *v4; // r12
  _QWORD *v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // [rsp+0h] [rbp-50h] BYREF
  __int64 v10; // [rsp+8h] [rbp-48h]
  const char *v11; // [rsp+10h] [rbp-40h] BYREF
  char v12; // [rsp+30h] [rbp-20h]
  char v13; // [rsp+31h] [rbp-1Fh]

  v1 = *(_BYTE *)(a1 + 296) == 0;
  *(_QWORD *)(a1 + 288) = *(_QWORD *)(a1 + 280);
  if ( v1 )
    *(_BYTE *)(a1 + 296) = 1;
  v9 = 0;
  v10 = 0;
  if ( !(unsigned __int8)sub_ECE2A0(a1, 9) )
  {
    v13 = 1;
    v11 = "unexpected token";
    v12 = 3;
    v2 = 1;
    if ( !(unsigned __int8)sub_EB61F0(a1, &v9) && v10 == 6 )
    {
      if ( *(_DWORD *)v9 != 1886218611 || (v2 = 0, *(_WORD *)(v9 + 4) != 25964) )
        v2 = 1;
    }
    if ( (unsigned __int8)sub_ECE0A0(a1, v2, &v11) || (unsigned __int8)sub_ECE000(a1) )
      return 1;
  }
  v4 = *(__int64 **)(a1 + 232);
  v5 = (_QWORD *)sub_ECD690(a1 + 40);
  sub_E9C600(v4, v10 != 0, v5, v6, v7, v8);
  return 0;
}
