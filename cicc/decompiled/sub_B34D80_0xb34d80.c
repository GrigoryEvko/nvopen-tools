// Function: sub_B34D80
// Address: 0xb34d80
//
__int64 __fastcall sub_B34D80(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v8; // r13
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // r12
  __int64 v14; // rax
  int v16; // eax
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  char v22; // [rsp+7h] [rbp-69h]
  char v23; // [rsp+7h] [rbp-69h]
  __int64 v24; // [rsp+8h] [rbp-68h]
  __int64 v25; // [rsp+8h] [rbp-68h]
  _QWORD v26[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v27[10]; // [rsp+20h] [rbp-50h] BYREF

  v8 = a5;
  v10 = a6;
  v11 = *(_QWORD *)(a3 + 8);
  if ( !a5 )
  {
    v16 = *(_DWORD *)(a2 + 32);
    v17 = *(_QWORD *)(a1 + 72);
    v22 = a4;
    v24 = v11;
    BYTE4(v27[0]) = *(_BYTE *)(a2 + 8) == 18;
    LODWORD(v27[0]) = v16;
    v18 = sub_BCB2A0(v17);
    v19 = sub_BCE1B0(v18, v27[0]);
    v20 = sub_AD62B0(v19);
    a4 = v22;
    v11 = v24;
    v8 = v20;
    if ( v10 )
      goto LABEL_3;
LABEL_5:
    v23 = a4;
    v25 = v11;
    v21 = sub_ACADE0((__int64 **)a2);
    a4 = v23;
    v11 = v25;
    v10 = v21;
    goto LABEL_3;
  }
  if ( !a6 )
    goto LABEL_5;
LABEL_3:
  v27[0] = a3;
  v12 = *(_QWORD *)(a1 + 72);
  v26[0] = a2;
  v26[1] = v11;
  v13 = (unsigned int)(1LL << a4);
  v14 = sub_BCB2D0(v12);
  v27[2] = v8;
  v27[3] = v10;
  v27[1] = sub_ACD640(v14, v13, 0);
  return sub_B34BE0(a1, 0xE3u, (int)v27, 4, (__int64)v26, 2, a7);
}
