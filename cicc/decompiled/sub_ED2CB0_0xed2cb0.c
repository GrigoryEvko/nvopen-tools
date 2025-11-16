// Function: sub_ED2CB0
// Address: 0xed2cb0
//
void __fastcall sub_ED2CB0(__int64 a1, char *a2, signed __int64 a3)
{
  __int64 v3; // rax
  _QWORD *v4; // r15
  __int64 v5; // rbx
  _QWORD *v6; // rax
  __int64 v7; // r13
  char v8; // al
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  char v12; // al
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // [rsp-80h] [rbp-80h]
  __int64 v18[2]; // [rsp-78h] [rbp-78h] BYREF
  _QWORD v19[2]; // [rsp-68h] [rbp-68h] BYREF
  __int64 v20; // [rsp-58h] [rbp-58h]
  __int64 v21; // [rsp-50h] [rbp-50h]
  __int64 v22; // [rsp-48h] [rbp-48h]

  if ( a3 )
  {
    v3 = sub_AC9B20(*(_QWORD *)a1, a2, a3, 1);
    BYTE4(v17) = 0;
    v4 = *(_QWORD **)(v3 + 8);
    v5 = v3;
    v18[0] = (__int64)"__llvm_profile_filename";
    LOWORD(v20) = 259;
    v6 = sub_BD2C40(88, unk_3F0FAE8);
    v7 = (__int64)v6;
    if ( v6 )
      sub_B30000((__int64)v6, a1, v4, 1, 4, v5, (__int64)v18, 0, 0, v17, 0);
    v8 = *(_BYTE *)(v7 + 32) & 0xCF | 0x10;
    *(_BYTE *)(v7 + 32) = v8;
    if ( (v8 & 0xF) != 9 )
      *(_BYTE *)(v7 + 33) |= 0x40u;
    v9 = *(_BYTE **)(a1 + 232);
    v10 = *(_QWORD *)(a1 + 240);
    v18[0] = (__int64)v19;
    sub_ED0570(v18, v9, (__int64)&v9[v10]);
    v11 = *(unsigned int *)(a1 + 284);
    v20 = *(_QWORD *)(a1 + 264);
    v21 = *(_QWORD *)(a1 + 272);
    v22 = *(_QWORD *)(a1 + 280);
    if ( (unsigned int)v11 > 8 || (v16 = 292, !_bittest64(&v16, v11)) )
    {
      v12 = *(_BYTE *)(v7 + 32);
      *(_BYTE *)(v7 + 32) = v12 & 0xF0;
      if ( (v12 & 0x30) != 0 )
        *(_BYTE *)(v7 + 33) |= 0x40u;
      v13 = sub_BAA410(a1, "__llvm_profile_filename", 0x17u);
      sub_B2F990(v7, v13, v14, v15);
    }
    if ( (_QWORD *)v18[0] != v19 )
      j_j___libc_free_0(v18[0], v19[0] + 1LL);
  }
}
