// Function: sub_38E2200
// Address: 0x38e2200
//
__int64 __fastcall sub_38E2200(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 v7; // r13
  __int16 v8; // r12
  int v9; // ebx
  __int64 v10; // rdx
  __int64 v11; // r9
  __int64 v12; // r11
  __int64 v13; // r10
  __int64 v14; // rdi
  __int64 result; // rax

  v2 = (__int64 *)a1[20];
  v3 = a1[23];
  v4 = a1[22];
  v5 = *v2;
  v6 = v2[1];
  v7 = v2[2];
  v8 = *((_WORD *)v2 + 12);
  v9 = *((_DWORD *)v2 + 7);
  v10 = v2[7];
  v11 = a1[21];
  v12 = v2[4];
  v13 = v2[5];
  v14 = v2[8];
  result = v2[6];
  *(_QWORD *)a2 = v5;
  *(_QWORD *)(a2 + 8) = v6;
  *(_QWORD *)(a2 + 16) = v7;
  *(_WORD *)(a2 + 24) = v8;
  *(_DWORD *)(a2 + 28) = v9;
  *(_QWORD *)(a2 + 32) = v12;
  *(_QWORD *)(a2 + 40) = v13;
  *(_QWORD *)(a2 + 48) = result;
  *(_QWORD *)(a2 + 56) = v10;
  *(_QWORD *)(a2 + 64) = v14;
  *(_QWORD *)(a2 + 72) = v11;
  *(_QWORD *)(a2 + 80) = v4;
  *(_QWORD *)(a2 + 88) = v3;
  *(_QWORD *)(a2 + 96) = v10;
  return result;
}
