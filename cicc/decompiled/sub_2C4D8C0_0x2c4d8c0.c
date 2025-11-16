// Function: sub_2C4D8C0
// Address: 0x2c4d8c0
//
__int64 __fastcall sub_2C4D8C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rdx
  unsigned __int64 v7; // rbx
  unsigned int v8; // eax
  unsigned __int64 v9; // rdx
  _QWORD v10[5]; // [rsp-28h] [rbp-28h] BYREF

  if ( !a1 )
    return 0;
  if ( sub_B46500((unsigned __int8 *)a1) )
    return 0;
  if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
    return 0;
  v2 = *(_QWORD *)(a1 + 16);
  if ( !v2 )
    return 0;
  if ( *(_QWORD *)(v2 + 8) )
    return 0;
  v4 = sub_B43CB0(a1);
  if ( (unsigned __int8)sub_B2D610(v4, 58) || (unsigned __int8)sub_D30ED0(a1) )
    return 0;
  v5 = *(_QWORD *)(a1 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
    v5 = **(_QWORD **)(v5 + 16);
  v10[0] = sub_BCAE30(v5);
  v10[1] = v6;
  v7 = sub_CA1930(v10);
  v8 = sub_DFB1F0(a2);
  if ( !v7 )
    return 0;
  if ( !v8 )
    return 0;
  v9 = v8 % v7;
  result = 1;
  if ( v7 & 7 | v9 )
    return 0;
  return result;
}
