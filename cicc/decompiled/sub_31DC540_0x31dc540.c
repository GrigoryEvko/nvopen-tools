// Function: sub_31DC540
// Address: 0x31dc540
//
__int64 __fastcall sub_31DC540(__int64 a1, __int64 a2)
{
  const char *v3; // rax
  __int64 v4; // rdx
  const char *v5; // rax
  __int64 v6; // rdx
  __int64 result; // rax
  const char *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  void (__fastcall *v11)(__int64, __int64, __int64, __int64); // rbx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r14
  void (__fastcall *v16)(__int64, unsigned __int64, _QWORD); // rbx
  unsigned __int64 v17; // rsi
  __int64 v18; // rbx
  __int64 *v19; // r12
  unsigned __int8 **v20; // rdx
  unsigned __int8 *v21; // rsi
  __int64 v22; // rdx
  unsigned __int8 *v23; // r14
  __int64 v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // r15
  __int64 *v27; // rbx
  __int64 v28; // rax
  __int64 v29; // rax
  void (__fastcall *v30)(__int64, __int64); // rbx
  __int64 v31; // rax
  __int64 v32; // r15
  const char *v33; // rax
  __int64 v34; // rdx
  const char *v35; // rax
  const char *v36; // rdx
  __int64 v37; // rax
  __int64 v38; // r14
  void (__fastcall *v39)(__int64, __int64, __int64, _QWORD); // rbx
  __int64 v40; // rax
  void (__fastcall *v41)(__int64 *, __int64); // [rsp+8h] [rbp-78h]
  __int64 *v42; // [rsp+10h] [rbp-70h]
  unsigned __int8 v43; // [rsp+18h] [rbp-68h]
  void (__fastcall *v44)(__int64 *, __int64); // [rsp+18h] [rbp-68h]
  __int64 v45; // [rsp+18h] [rbp-68h]
  __int64 v46; // [rsp+18h] [rbp-68h]
  const char *v47[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v48; // [rsp+40h] [rbp-40h]

  v3 = sub_BD5D20(a2);
  if ( v4 == 9 && (v4 = 0x6573752E6D766C6CLL, *(_QWORD *)v3 == 0x6573752E6D766C6CLL) && v3[8] == 100 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 208) + 292LL) )
    {
      v43 = *(_BYTE *)(*(_QWORD *)(a1 + 208) + 292LL);
      sub_31DC490(a1, *(_QWORD *)(a2 - 32));
      return v43;
    }
  }
  else if ( (*(_BYTE *)(a2 + 35) & 4) == 0
         || (v13 = sub_B31D10(a2, a2, v4), v14 != 13)
         || *(_QWORD *)v13 != 0x74656D2E6D766C6CLL
         || *(_DWORD *)(v13 + 8) != 1952539745
         || *(_BYTE *)(v13 + 12) != 97 )
  {
    if ( (*(_BYTE *)(a2 + 32) & 0xF) != 1 )
    {
      v5 = sub_BD5D20(a2);
      if ( v6 == 22
        && !(*(_QWORD *)v5 ^ 0x6D72612E6D766C6CLL | *((_QWORD *)v5 + 1) ^ 0x6D79732E63653436LL)
        && *((_DWORD *)v5 + 4) == 1835822946
        && *((_WORD *)v5 + 10) == 28769 )
      {
        v15 = *(_QWORD *)(a1 + 224);
        v16 = *(void (__fastcall **)(__int64, unsigned __int64, _QWORD))(*(_QWORD *)v15 + 176LL);
        v17 = sub_E6E280(*(_QWORD **)(a1 + 216), ".hybmp$x", 8u, 0x200u);
        v16(v15, v17, 0);
        v18 = *(_QWORD *)(a2 - 32);
        if ( (*(_BYTE *)(v18 + 7) & 0x40) != 0 )
        {
          v19 = *(__int64 **)(v18 - 8);
          v42 = &v19[4 * (*(_DWORD *)(v18 + 4) & 0x7FFFFFF)];
        }
        else
        {
          v42 = *(__int64 **)(a2 - 32);
          v19 = (__int64 *)(v18 - 32LL * (*(_DWORD *)(v18 + 4) & 0x7FFFFFF));
        }
        while ( v42 != v19 )
        {
          v32 = *v19;
          if ( (*(_BYTE *)(*v19 + 7) & 0x40) != 0 )
            v20 = *(unsigned __int8 ***)(v32 - 8);
          else
            v20 = (unsigned __int8 **)(v32 - 32LL * (*(_DWORD *)(v32 + 4) & 0x7FFFFFF));
          v21 = sub_BD3990(*v20, v17);
          if ( (*(_BYTE *)(v32 + 7) & 0x40) != 0 )
            v22 = *(_QWORD *)(v32 - 8);
          else
            v22 = v32 - 32LL * (*(_DWORD *)(v32 + 4) & 0x7FFFFFF);
          v23 = sub_BD3990(*(unsigned __int8 **)(v22 + 32), (__int64)v21);
          if ( (*(_BYTE *)(v32 + 7) & 0x40) != 0 )
            v24 = *(_QWORD *)(v32 - 8);
          else
            v24 = v32 - 32LL * (*(_DWORD *)(v32 + 4) & 0x7FFFFFF);
          v25 = *(_QWORD *)(v24 + 64);
          v26 = *(_QWORD **)(v25 + 24);
          if ( *(_DWORD *)(v25 + 32) > 0x40u )
            v26 = (_QWORD *)*v26;
          v27 = *(__int64 **)(a1 + 224);
          v28 = *v27;
          if ( (v21[33] & 3) == 1 )
          {
            v41 = *(void (__fastcall **)(__int64 *, __int64))(v28 + 352);
            v46 = *(_QWORD *)(a1 + 216);
            v35 = sub_BD5D20((__int64)v21);
            v48 = 1283;
            v47[0] = "__imp_";
            v47[2] = v35;
            v47[3] = v36;
            v37 = sub_E6C460(v46, v47);
            v41(v27, v37);
          }
          else
          {
            v44 = *(void (__fastcall **)(__int64 *, __int64))(v28 + 352);
            v29 = sub_31DB510(a1, (__int64)v21);
            v44(v27, v29);
          }
          v19 += 4;
          v45 = *(_QWORD *)(a1 + 224);
          v30 = *(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v45 + 352LL);
          v31 = sub_31DB510(a1, (__int64)v23);
          v30(v45, v31);
          v17 = (int)v26;
          (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 224) + 536LL))(
            *(_QWORD *)(a1 + 224),
            (int)v26,
            4);
        }
      }
      else
      {
        result = 0;
        if ( (*(_BYTE *)(a2 + 32) & 0xF) != 6 )
          return result;
        v8 = sub_BD5D20(a2);
        if ( v9 == 17
          && !(*(_QWORD *)v8 ^ 0x6F6C672E6D766C6CLL | *((_QWORD *)v8 + 1) ^ 0x726F74635F6C6162LL)
          && v8[16] == 115 )
        {
          v10 = *(_QWORD *)(a2 - 32);
          v11 = *(void (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1 + 232LL);
          v12 = sub_B2F730(a2);
          v11(a1, v12, v10, 1);
          return 1;
        }
        v33 = sub_BD5D20(a2);
        if ( v34 != 17
          || *(_QWORD *)v33 ^ 0x6F6C672E6D766C6CLL | *((_QWORD *)v33 + 1) ^ 0x726F74645F6C6162LL
          || v33[16] != 115 )
        {
          sub_C64ED0("unknown special variable with appending linkage", 1u);
        }
        v38 = *(_QWORD *)(a2 - 32);
        v39 = *(void (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)a1 + 232LL);
        v40 = sub_B2F730(a2);
        v39(a1, v40, v38, 0);
      }
    }
  }
  return 1;
}
