// Function: sub_2145E30
// Address: 0x2145e30
//
__int64 __fastcall sub_2145E30(__int64 a1, __int64 a2)
{
  unsigned __int64 *v2; // rax
  __int64 v3; // r13
  unsigned __int64 v4; // r14
  __int64 v5; // rax
  char v6; // dl
  __int64 v7; // rax
  bool v8; // al
  __int64 v9; // rdx
  _QWORD *v10; // rax
  __int64 v12; // [rsp+0h] [rbp-50h] BYREF
  __int64 v13; // [rsp+8h] [rbp-48h]
  __int64 v14; // [rsp+10h] [rbp-40h] BYREF
  __int64 v15; // [rsp+18h] [rbp-38h]
  _BYTE v16[8]; // [rsp+20h] [rbp-30h] BYREF
  __int64 v17; // [rsp+28h] [rbp-28h]

  v2 = *(unsigned __int64 **)(a2 + 32);
  LODWORD(v13) = 0;
  LODWORD(v15) = 0;
  v12 = 0;
  v3 = v2[1];
  v14 = 0;
  v4 = *v2;
  v5 = *(_QWORD *)(*v2 + 40) + 16LL * *((unsigned int *)v2 + 2);
  v6 = *(_BYTE *)v5;
  v7 = *(_QWORD *)(v5 + 8);
  v16[0] = v6;
  v17 = v7;
  if ( v6 )
    v8 = (unsigned __int8)(v6 - 14) <= 0x47u || (unsigned __int8)(v6 - 2) <= 5u;
  else
    v8 = sub_1F58CF0((__int64)v16);
  if ( v8 )
    sub_20174B0(a1, v4, v3, &v12, &v14);
  else
    sub_2016B80(a1, v4, v3, &v12, &v14);
  v9 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 88LL);
  v10 = *(_QWORD **)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v10 = (_QWORD *)*v10;
  if ( v10 )
    return v14;
  else
    return v12;
}
