// Function: sub_31DF3C0
// Address: 0x31df3c0
//
char __fastcall sub_31DF3C0(__int64 a1)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  int v9; // r8d
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned int v12; // r15d
  unsigned int v13; // ecx
  __int64 v14; // rax
  _QWORD *v15; // rdi
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // r8
  __int64 v20; // r12
  __int64 v21; // r11
  void (__fastcall *v22)(__int64, unsigned __int64, _QWORD); // r10
  unsigned __int64 v23; // rax
  unsigned int v24; // esi
  unsigned __int64 v25; // rax
  int v26; // edx
  int v28; // [rsp+8h] [rbp-98h]
  void (__fastcall *v29)(__int64, unsigned __int64, _QWORD); // [rsp+8h] [rbp-98h]
  __int64 v30[4]; // [rsp+10h] [rbp-90h] BYREF
  char v31; // [rsp+30h] [rbp-70h]
  char v32; // [rsp+31h] [rbp-6Fh]
  unsigned __int64 v33[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v34; // [rsp+60h] [rbp-40h]

  v2 = **(_QWORD **)(a1 + 232);
  v30[0] = sub_B2D7E0(v2, "patchable-function-prefix", 0x19u);
  v3 = sub_A72240(v30);
  if ( sub_C93C90(v3, v4, 0xAu, v33) || (v5 = v33[0], v33[0] != LODWORD(v33[0])) )
  {
    v30[0] = sub_B2D7E0(v2, "patchable-function-entry", 0x18u);
    v10 = sub_A72240(v30);
    LOBYTE(v8) = sub_C93C90(v10, v11, 0xAu, v33);
    if ( (_BYTE)v8 )
      return v8;
    LODWORD(v8) = v33[0];
    if ( v33[0] != LODWORD(v33[0]) )
      return v8;
    v9 = 0;
  }
  else
  {
    v28 = v33[0];
    v30[0] = sub_B2D7E0(v2, "patchable-function-entry", 0x18u);
    v6 = sub_A72240(v30);
    LOBYTE(v8) = sub_C93C90(v6, v7, 0xAu, v33);
    v9 = v28;
    if ( (_BYTE)v8 || (LODWORD(v8) = v33[0], v33[0] != LODWORD(v33[0])) )
    {
      if ( !v5 )
        return v8;
      goto LABEL_11;
    }
  }
  if ( !((unsigned int)v8 | v9) )
    return v8;
LABEL_11:
  v12 = sub_31DAFE0(a1);
  v8 = *(_QWORD *)(a1 + 200);
  v13 = *(_DWORD *)(v8 + 564);
  if ( v13 == 3 )
  {
    v14 = *(_QWORD *)(a1 + 208);
    v15 = *(_QWORD **)(v2 + 48);
    if ( *(_BYTE *)(v14 + 392) || (v26 = *(_DWORD *)(v14 + 384), v26 > 1) && (v26 != 2 || *(int *)(v14 + 388) > 35) )
    {
      if ( v15 )
      {
        v16 = sub_AA8810(v15);
        v18 = *(_QWORD *)(v2 + 48);
        v13 = 643;
        v15 = (_QWORD *)v16;
      }
      else
      {
        v18 = 0;
        v17 = 0;
        v13 = 131;
      }
      v19 = *(_QWORD *)(a1 + 280);
    }
    else
    {
      v18 = *(_QWORD *)(v2 + 48);
      v17 = 0;
      v15 = 0;
      v19 = 0;
    }
    v20 = *(_QWORD *)(a1 + 224);
    v21 = *(_QWORD *)(a1 + 216);
    v22 = *(void (__fastcall **)(__int64, unsigned __int64, _QWORD))(*(_QWORD *)v20 + 176LL);
    v34 = 261;
    v30[0] = (__int64)"__patchable_function_entries";
    v33[0] = (unsigned __int64)v15;
    v33[1] = v17;
    v29 = v22;
    v32 = 1;
    v31 = 3;
    v23 = sub_E71CB0(v21, (size_t *)v30, 1, v13, 0, (__int64)v33, v18 != 0, -1, v19);
    v29(v20, v23, 0);
    v24 = -1;
    if ( v12 )
    {
      _BitScanReverse64(&v25, v12);
      v24 = 63 - (v25 ^ 0x3F);
    }
    sub_31DCA70(a1, v24, 0, 0);
    LOBYTE(v8) = (unsigned __int8)sub_E9A500(*(_QWORD *)(a1 + 224), *(_QWORD *)(a1 + 272), v12, 0);
  }
  return v8;
}
