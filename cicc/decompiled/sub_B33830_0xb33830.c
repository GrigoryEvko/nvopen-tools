// Function: sub_B33830
// Address: 0xb33830
//
__int64 __fastcall sub_B33830(__int64 a1, char *a2, signed __int64 a3, __int64 a4, int a5, __int64 a6, char a7)
{
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r12
  _QWORD *v14; // [rsp+8h] [rbp-48h]
  __int64 v15; // [rsp+18h] [rbp-38h]

  v10 = sub_AC9B20(*(_QWORD *)(a1 + 72), a2, a3, a7);
  if ( !a6 )
    a6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 72LL) + 40LL);
  LODWORD(v15) = a5;
  BYTE4(v15) = 1;
  v14 = *(_QWORD **)(v10 + 8);
  v11 = sub_BD2C40(88, unk_3F0FAE8);
  v12 = v11;
  if ( v11 )
    sub_B30000(v11, a6, v14, 1, 8, v10, a4, 0, 0, v15, 0);
  *(_BYTE *)(v12 + 32) = *(_BYTE *)(v12 + 32) & 0x3F | 0x80;
  sub_B2F770(v12, 0);
  return v12;
}
