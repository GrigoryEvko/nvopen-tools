// Function: sub_15643F0
// Address: 0x15643f0
//
__int64 __fastcall sub_15643F0(__int64 a1, unsigned int a2, _QWORD *a3)
{
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 result; // rax
  __int64 v9; // rdx
  _QWORD v10[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v11[2]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v12; // [rsp+20h] [rbp-30h]

  v4 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL) + 8LL);
  v5 = sub_15E0530(a1);
  v6 = sub_16432A0(v5);
  v7 = sub_16463B0(v6, 4);
  result = 0;
  if ( v4 == v7 )
  {
    v10[0] = sub_1649960(a1);
    v12 = 773;
    v11[0] = v10;
    v10[1] = v9;
    v11[1] = ".old";
    sub_164B780(a1, v11);
    *a3 = sub_15E26F0(*(_QWORD *)(a1 + 40), a2, 0, 0);
    return 1;
  }
  return result;
}
