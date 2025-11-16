// Function: sub_2C1B700
// Address: 0x2c1b700
//
__int64 __fastcall sub_2C1B700(__int64 a1)
{
  unsigned __int8 *v1; // r13
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // r12
  __int64 v6; // r15
  int v7; // ebx
  __int64 v9; // rax
  __int64 v10; // r9
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // [rsp+0h] [rbp-50h] BYREF
  __int64 v14; // [rsp+8h] [rbp-48h] BYREF
  __int64 v15; // [rsp+10h] [rbp-40h] BYREF
  __int64 v16[7]; // [rsp+18h] [rbp-38h] BYREF

  v1 = *(unsigned __int8 **)(a1 + 136);
  v2 = **(_QWORD **)(a1 + 48);
  if ( !v1 )
  {
    v9 = sub_22077B0(0xB0u);
    v5 = v9;
    if ( !v9 )
      return v5;
    v6 = *(_QWORD *)(a1 + 168);
    v13 = 0;
    v14 = 0;
    v7 = *(_DWORD *)(a1 + 160);
    v15 = v2;
    v16[0] = 0;
    sub_2AAF310(v9, 16, &v15, 1, v16, v10);
    sub_9C6650(v16);
    sub_2BF0340(v5 + 96, 1, 0, v5, v11, v12);
    *(_QWORD *)v5 = &unk_4A231C8;
    *(_QWORD *)(v5 + 40) = &unk_4A23200;
    *(_QWORD *)(v5 + 96) = &unk_4A23238;
    sub_9C6650(&v14);
    *(_BYTE *)(v5 + 152) = 7;
    *(_DWORD *)(v5 + 156) = 0;
    *(_QWORD *)v5 = &unk_4A23258;
    *(_QWORD *)(v5 + 40) = &unk_4A23290;
    *(_QWORD *)(v5 + 96) = &unk_4A232C8;
    sub_9C6650(&v13);
    goto LABEL_4;
  }
  v3 = sub_22077B0(0xB0u);
  v5 = v3;
  if ( v3 )
  {
    v6 = *(_QWORD *)(a1 + 168);
    v7 = *(_DWORD *)(a1 + 160);
    sub_2ABA9E0(v3, 16, v2, v1, v4);
LABEL_4:
    *(_DWORD *)(v5 + 160) = v7;
    *(_QWORD *)(v5 + 168) = v6;
    *(_QWORD *)v5 = &unk_4A23F58;
    *(_QWORD *)(v5 + 40) = &unk_4A23F90;
    *(_QWORD *)(v5 + 96) = &unk_4A23FC8;
  }
  return v5;
}
