// Function: sub_2425180
// Address: 0x2425180
//
__int64 __fastcall sub_2425180(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  _QWORD *v4; // rdi
  __int64 v5; // rax
  _QWORD *v6; // rdi
  __int64 v7; // rax
  _QWORD *v8; // rdi
  __int64 *v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // r12
  __int64 *v13; // r15
  unsigned __int64 v14; // r8
  int v15; // ebx
  unsigned __int64 v17; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v18[10]; // [rsp+10h] [rbp-50h] BYREF

  v3 = sub_BCE3C0(*(__int64 **)(a1 + 168), 0);
  v4 = *(_QWORD **)(a1 + 168);
  v18[0] = v3;
  v5 = sub_BCB2D0(v4);
  v6 = *(_QWORD **)(a1 + 168);
  v18[1] = v5;
  v7 = sub_BCB2D0(v6);
  v8 = *(_QWORD **)(a1 + 168);
  v18[2] = v7;
  v9 = (__int64 *)sub_BCB120(v8);
  v10 = sub_BCF480(v9, v18, 3, 0);
  v17 = 0;
  v11 = *(_QWORD *)(a1 + 128);
  v12 = v10;
  v13 = *(__int64 **)(a1 + 168);
  if ( *(_BYTE *)(*(_QWORD *)a2 + 168LL) )
  {
    v15 = 79;
    goto LABEL_5;
  }
  v14 = 0;
  v15 = 54;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 170LL) )
  {
LABEL_5:
    v17 = sub_A7A090((__int64 *)&v17, v13, 2, v15);
    v14 = sub_A7A090((__int64 *)&v17, v13, 3, v15);
  }
  return sub_BA8C10(v11, (__int64)"llvm_gcda_start_file", 0x14u, v12, v14);
}
