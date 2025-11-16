// Function: sub_2BF0910
// Address: 0x2bf0910
//
__int64 __fastcall sub_2BF0910(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  __int64 v3; // rbx
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v9[4]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v10; // [rsp+20h] [rbp-30h]

  v2 = *(_QWORD *)(a2 + 104);
  v3 = *(_QWORD *)(a2 + 112);
  v9[0] = a1 + 16;
  v10 = 260;
  v4 = *(_QWORD *)(v2 + 72);
  v5 = sub_AA48A0(v2);
  v6 = sub_22077B0(0x50u);
  v7 = v6;
  if ( v6 )
    sub_AA4D50(v6, v5, (__int64)v9, v4, v3);
  return v7;
}
