// Function: sub_2FC95F0
// Address: 0x2fc95f0
//
__int64 __fastcall sub_2FC95F0(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 *v11; // rdi
  __int64 v12; // r14
  __int64 v13; // rax
  char v14; // bl
  _QWORD *v15; // rax
  __int64 v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rdx
  unsigned int v19; // esi
  const char *v20; // [rsp+0h] [rbp-90h] BYREF
  char v21; // [rsp+20h] [rbp-70h]
  char v22; // [rsp+21h] [rbp-6Fh]
  _BYTE v23[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v24; // [rsp+50h] [rbp-40h]

  v6 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 920LL))(a1, a3);
  v7 = sub_BAA870(a2);
  if ( v8 == 3 )
  {
    if ( *(_WORD *)v7 != 27764 || *(_BYTE *)(v7 + 2) != 115 )
      goto LABEL_3;
  }
  else if ( v8 )
  {
    goto LABEL_3;
  }
  if ( v6 )
  {
    v11 = *(__int64 **)(a3 + 72);
    v22 = 1;
    v20 = "StackGuard";
    v21 = 3;
    v12 = sub_BCE3C0(v11, 0);
    v13 = sub_AA4E30(*(_QWORD *)(a3 + 48));
    v14 = sub_AE5020(v13, v12);
    v24 = 257;
    v15 = sub_BD2C40(80, 1u);
    v9 = (__int64)v15;
    if ( v15 )
      sub_B4D190((__int64)v15, v12, v6, (__int64)v23, 1, v14, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, const char **, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
      *(_QWORD *)(a3 + 88),
      v9,
      &v20,
      *(_QWORD *)(a3 + 56),
      *(_QWORD *)(a3 + 64));
    v16 = *(_QWORD *)a3;
    v17 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
    while ( v17 != v16 )
    {
      v18 = *(_QWORD *)(v16 + 8);
      v19 = *(_DWORD *)v16;
      v16 += 16;
      sub_B99FD0(v9, v19, v18);
    }
    return v9;
  }
LABEL_3:
  if ( a4 )
    *a4 = 1;
  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 928LL))(a1, a2);
  v24 = 257;
  HIDWORD(v20) = 0;
  return sub_B33D10(a3, 0x154u, 0, 0, 0, 0, (unsigned int)v20, (__int64)v23);
}
