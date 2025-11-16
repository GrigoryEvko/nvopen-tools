// Function: sub_2EB42A0
// Address: 0x2eb42a0
//
__int64 __fastcall sub_2EB42A0(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 v8; // r13
  const char *v9; // rax
  size_t v10; // rdx
  _BYTE *v11; // rdi
  unsigned __int8 *v12; // rsi
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  size_t v18; // [rsp+8h] [rbp-38h]

  v8 = sub_904010(*a2, "MachinePostDominatorTree for machine function: ");
  v9 = sub_2E791E0(a3);
  v11 = *(_BYTE **)(v8 + 32);
  v12 = (unsigned __int8 *)v9;
  v13 = *(_QWORD *)(v8 + 24);
  if ( v13 - (unsigned __int64)v11 < v10 )
  {
    v17 = sub_CB6200(v8, v12, v10);
    v11 = *(_BYTE **)(v17 + 32);
    v8 = v17;
    v13 = *(_QWORD *)(v17 + 24);
  }
  else if ( v10 )
  {
    v18 = v10;
    memcpy(v11, v12, v10);
    v16 = *(_QWORD *)(v8 + 24);
    v11 = (_BYTE *)(*(_QWORD *)(v8 + 32) + v18);
    *(_QWORD *)(v8 + 32) = v11;
    if ( v16 > (unsigned __int64)v11 )
      goto LABEL_4;
LABEL_7:
    sub_CB5D20(v8, 10);
    goto LABEL_5;
  }
  if ( v13 <= (unsigned __int64)v11 )
    goto LABEL_7;
LABEL_4:
  *(_QWORD *)(v8 + 32) = v11 + 1;
  *v11 = 10;
LABEL_5:
  v14 = sub_2EB2140(a4, qword_50209E0, (__int64)a3);
  sub_2EB4190(v14 + 8, *a2);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
