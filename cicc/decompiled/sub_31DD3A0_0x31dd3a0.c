// Function: sub_31DD3A0
// Address: 0x31dd3a0
//
__int64 __fastcall sub_31DD3A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  _QWORD *v5; // rdi
  __int64 v7; // rdx
  const char *v8; // rsi
  unsigned __int64 v9; // rsi
  __int64 v10; // r14
  __int64 v11; // r13
  __int64 v12; // r15
  __int64 v13; // r12
  _QWORD *v14; // r14
  unsigned int v15; // esi
  unsigned int v16; // r13d
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 (__fastcall *v19)(__int64, __int64, _QWORD); // r13
  unsigned int v20; // eax
  __int64 v21; // rax
  __int64 v22; // r10
  char v23; // cl
  _QWORD *v24; // rdi
  _QWORD *v25; // r13
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // [rsp+0h] [rbp-A0h]
  size_t v30[4]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v31; // [rsp+30h] [rbp-70h]
  _QWORD v32[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v33; // [rsp+60h] [rbp-40h]

  result = *(_QWORD *)(a2 + 16);
  if ( *(_QWORD *)(a2 + 8) != result )
  {
    v5 = *(_QWORD **)(a3 + 48);
    v7 = 0;
    v8 = byte_3F871B3;
    if ( v5 )
      v8 = (const char *)sub_AA8810(v5);
    result = *(unsigned int *)(*(_QWORD *)(a1 + 200) + 564LL);
    if ( (*(_DWORD *)(*(_QWORD *)(a1 + 200) + 564LL) & 0xFFFFFFFD) == 1 )
    {
      if ( (_DWORD)result == 3 )
      {
        v32[0] = v8;
        v21 = *(_QWORD *)(a1 + 280);
        v22 = *(_QWORD *)(a1 + 216);
        v23 = *(_BYTE *)(v21 + 9);
        v32[1] = v7;
        v33 = 261;
        v31 = 261;
        v30[0] = (size_t)".llvm_jump_table_sizes";
        v30[1] = 22;
        if ( (v23 & 7) != 2 )
          v21 = 0;
        v9 = sub_E71CB0(
               v22,
               v30,
               1879002125,
               (unsigned __int8)(*(_QWORD *)(a3 + 48) != 0) << 9,
               0,
               (__int64)v32,
               *(_QWORD *)(a3 + 48) != 0,
               -1,
               v21);
      }
      else
      {
        v9 = 0;
        if ( (_DWORD)result == 1 )
        {
          v24 = *(_QWORD **)(a3 + 48);
          v25 = *(_QWORD **)(a1 + 216);
          if ( v24 )
          {
            v26 = sub_AA8810(v24);
            v9 = sub_E6DEB0(v25, ".llvm_jump_table_sizes", 0x16u, 0x42001040u, v26, v27, 5u, 0xFFFFFFFF);
          }
          else
          {
            v9 = sub_E6E280(*(_QWORD **)(a1 + 216), ".llvm_jump_table_sizes", 0x16u, 0x42000040u);
          }
        }
      }
      (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 176LL))(
        *(_QWORD *)(a1 + 224),
        v9,
        0);
      v10 = *(_QWORD *)(a2 + 8);
      result = *(_QWORD *)(a2 + 16);
      v11 = (result - v10) >> 5;
      if ( (_DWORD)v11 )
      {
        v12 = 0;
        v28 = (unsigned int)v11;
        while ( 1 )
        {
          v13 = *(_QWORD *)(a1 + 224);
          v14 = (_QWORD *)(32 * v12 + v10);
          v15 = v12;
          v16 = sub_AE4380(*(_QWORD *)(a1 + 200) + 16LL, *(_DWORD *)(*(_QWORD *)(a1 + 200) + 24LL));
          ++v12;
          v17 = sub_31DD380(a1, v15, 0);
          sub_E9A500(v13, v17, v16, 0);
          v18 = *(_QWORD *)(a1 + 224);
          v19 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v18 + 536LL);
          v20 = sub_AE4380(*(_QWORD *)(a1 + 200) + 16LL, *(_DWORD *)(*(_QWORD *)(a1 + 200) + 24LL));
          result = v19(v18, (__int64)(v14[1] - *v14) >> 3, v20);
          if ( v12 == v28 )
            break;
          v10 = *(_QWORD *)(a2 + 8);
        }
      }
    }
  }
  return result;
}
