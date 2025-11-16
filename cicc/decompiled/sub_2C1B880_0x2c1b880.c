// Function: sub_2C1B880
// Address: 0x2c1b880
//
__int64 __fastcall sub_2C1B880(__int64 a1)
{
  unsigned __int8 *v1; // r13
  _QWORD *v2; // rdx
  __int64 v3; // r15
  __int64 v4; // rax
  __int64 v5; // r12
  int v6; // ebx
  __int64 v8; // [rsp+0h] [rbp-50h]
  _QWORD *v9; // [rsp+8h] [rbp-48h]
  _QWORD *v10; // [rsp+8h] [rbp-48h]
  __int64 v11[7]; // [rsp+18h] [rbp-38h] BYREF

  v1 = *(unsigned __int8 **)(a1 + 136);
  v2 = *(_QWORD **)(a1 + 48);
  v3 = *(unsigned int *)(a1 + 56);
  v11[0] = *(_QWORD *)(a1 + 88);
  if ( v11[0] )
  {
    v9 = v2;
    sub_2AAAFA0(v11);
    v2 = v9;
  }
  v10 = v2;
  v4 = sub_22077B0(0xB8u);
  v5 = v4;
  if ( v4 )
  {
    v6 = *(_DWORD *)(a1 + 160);
    v8 = *(_QWORD *)(a1 + 168);
    sub_2ABAD10(v4, 18, v10, v3, v1, v8);
    *(_DWORD *)(v5 + 160) = v6;
    *(_QWORD *)(v5 + 168) = v8;
    *(_QWORD *)v5 = &unk_4A23D28;
    *(_QWORD *)(v5 + 40) = &unk_4A23D68;
    *(_QWORD *)(v5 + 96) = &unk_4A23DA0;
    *(_BYTE *)(v5 + 176) = sub_B46420((__int64)v1);
    *(_BYTE *)(v5 + 177) = sub_B46490((__int64)v1);
    *(_BYTE *)(v5 + 178) = sub_B46970(v1);
  }
  sub_9C6650(v11);
  return v5;
}
