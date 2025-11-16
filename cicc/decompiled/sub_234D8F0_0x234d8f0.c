// Function: sub_234D8F0
// Address: 0x234d8f0
//
__int64 __fastcall sub_234D8F0(__int64 a1, int *a2, char a3, char a4)
{
  __int64 v4; // rax
  __int64 v5; // r12
  unsigned int v6; // ebx
  int v7; // r14d
  __int16 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v12; // [rsp+0h] [rbp-40h]

  v4 = *((_QWORD *)a2 + 3);
  v5 = *((_QWORD *)a2 + 2);
  ++*((_QWORD *)a2 + 1);
  v6 = a2[8];
  *((_QWORD *)a2 + 2) = 0;
  v7 = *a2;
  *((_QWORD *)a2 + 3) = 0;
  v8 = *((_WORD *)a2 + 2);
  a2[8] = 0;
  v12 = v4;
  v9 = sub_22077B0(0x30u);
  if ( v9 )
  {
    *(_DWORD *)(v9 + 40) = v6;
    *(_DWORD *)(v9 + 8) = v7;
    *(_WORD *)(v9 + 12) = v8;
    *(_QWORD *)(v9 + 16) = 1;
    *(_QWORD *)(v9 + 32) = v12;
    *(_QWORD *)v9 = &unk_4A11838;
    v10 = 0;
    *(_QWORD *)(v9 + 24) = v5;
    v5 = 0;
  }
  else
  {
    v10 = 4LL * v6;
  }
  *(_QWORD *)a1 = v9;
  *(_BYTE *)(a1 + 8) = a3;
  *(_BYTE *)(a1 + 9) = a4;
  sub_C7D6A0(v5, v10, 4);
  return a1;
}
