// Function: sub_234BD10
// Address: 0x234bd10
//
__int64 __fastcall sub_234BD10(__int64 a1, int *a2, char a3)
{
  __int64 v3; // rax
  __int64 v4; // r12
  unsigned int v5; // ebx
  int v6; // r14d
  __int16 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v11; // [rsp+0h] [rbp-40h]

  v3 = *((_QWORD *)a2 + 3);
  v4 = *((_QWORD *)a2 + 2);
  v5 = a2[8];
  ++*((_QWORD *)a2 + 1);
  *((_QWORD *)a2 + 2) = 0;
  v6 = *a2;
  *((_QWORD *)a2 + 3) = 0;
  v7 = *((_WORD *)a2 + 2);
  a2[8] = 0;
  v11 = v3;
  v8 = sub_22077B0(0x30u);
  if ( v8 )
  {
    *(_DWORD *)(v8 + 8) = v6;
    v9 = 0;
    *(_WORD *)(v8 + 12) = v7;
    *(_QWORD *)(v8 + 16) = 1;
    *(_QWORD *)v8 = &unk_4A11838;
    *(_DWORD *)(v8 + 40) = v5;
    *(_QWORD *)(v8 + 32) = v11;
    *(_QWORD *)(v8 + 24) = v4;
    v4 = 0;
  }
  else
  {
    v9 = 4LL * v5;
  }
  *(_QWORD *)a1 = v8;
  *(_BYTE *)(a1 + 8) = a3;
  sub_C7D6A0(v4, v9, 4);
  return a1;
}
