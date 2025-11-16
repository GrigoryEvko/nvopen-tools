// Function: sub_2353AC0
// Address: 0x2353ac0
//
__int64 __fastcall sub_2353AC0(unsigned __int64 *a1, int *a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 v4; // r13
  __int16 v5; // ax
  int v6; // r15d
  __int64 v7; // rax
  __int64 v8; // r13
  __int16 v10; // [rsp+Eh] [rbp-42h]
  unsigned __int64 v11[7]; // [rsp+18h] [rbp-38h] BYREF

  ++*((_QWORD *)a2 + 1);
  v2 = *((_QWORD *)a2 + 2);
  v3 = (unsigned int)a2[8];
  v4 = *((_QWORD *)a2 + 3);
  *((_QWORD *)a2 + 2) = 0;
  v5 = *((_WORD *)a2 + 2);
  v6 = *a2;
  *((_QWORD *)a2 + 3) = 0;
  a2[8] = 0;
  v10 = v5;
  v7 = sub_22077B0(0x30u);
  if ( v7 )
  {
    *(_DWORD *)(v7 + 8) = v6;
    *(_QWORD *)(v7 + 16) = 1;
    *(_WORD *)(v7 + 12) = v10;
    *(_QWORD *)v7 = &unk_4A11838;
    *(_DWORD *)(v7 + 40) = v3;
    *(_QWORD *)(v7 + 24) = v2;
    v2 = 0;
    *(_QWORD *)(v7 + 32) = v4;
    v8 = 0;
  }
  else
  {
    v8 = 4 * v3;
  }
  v11[0] = v7;
  sub_2353900(a1, v11);
  if ( v11[0] )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v11[0] + 8LL))(v11[0]);
  return sub_C7D6A0(v2, v8, 4);
}
