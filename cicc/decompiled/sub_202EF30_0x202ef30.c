// Function: sub_202EF30
// Address: 0x202ef30
//
__int64 __fastcall sub_202EF30(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rdx
  __int64 v4; // r12
  unsigned __int8 *v5; // rax
  unsigned int v6; // r14d
  __int64 (__fastcall *v7)(__int64, __int64, __int64, _QWORD, __int64); // r15
  __int64 v8; // rax
  unsigned __int8 v9; // r14
  unsigned __int8 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v14; // [rsp+8h] [rbp-58h]
  _BYTE v15[16]; // [rsp+10h] [rbp-50h] BYREF

  v2 = *a1;
  v3 = a1[1];
  v4 = *(_QWORD *)(v3 + 48);
  v5 = (unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL)
                         + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL));
  v6 = *v5;
  v14 = *((_QWORD *)v5 + 1);
  v7 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)*a1 + 264LL);
  v8 = sub_1E0A0C0(*(_QWORD *)(v3 + 32));
  v9 = v7(v2, v8, v4, v6, v14);
  v10 = v9;
  v12 = v11;
  sub_1F40D10((__int64)v15, *a1, *(_QWORD *)(a1[1] + 48), v9, v11);
  if ( v15[0] == 7 )
  {
    sub_1F40D10((__int64)v15, *a1, v4, v9, v12);
    return v15[8];
  }
  return v10;
}
