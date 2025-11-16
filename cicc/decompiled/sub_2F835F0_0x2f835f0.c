// Function: sub_2F835F0
// Address: 0x2f835f0
//
__int64 __fastcall sub_2F835F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // r15
  __int64 v8; // r14
  __int64 v9; // rax
  char v10; // bl
  _QWORD *v11; // rax
  __int64 v12; // r12
  __int64 v13; // rbx
  __int64 v14; // r13
  __int64 v15; // rdx
  unsigned int v16; // esi
  const char *v18; // [rsp+0h] [rbp-90h] BYREF
  char v19; // [rsp+20h] [rbp-70h]
  char v20; // [rsp+21h] [rbp-6Fh]
  _BYTE v21[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v22; // [rsp+50h] [rbp-40h]

  v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 920LL))(*(_QWORD *)(a1 + 8));
  if ( v5 )
  {
    v6 = *(_QWORD *)(a2 + 48);
    v7 = *(_QWORD *)(a1 + 40);
    v8 = v5;
    v20 = 1;
    v19 = 3;
    v18 = "StackGuard";
    v9 = sub_AA4E30(v6);
    v10 = sub_AE5020(v9, v7);
    v22 = 257;
    v11 = sub_BD2C40(80, 1u);
    v12 = (__int64)v11;
    if ( v11 )
      sub_B4D190((__int64)v11, v7, v8, (__int64)v21, 0, v10, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, const char **, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v12,
      &v18,
      *(_QWORD *)(a2 + 56),
      *(_QWORD *)(a2 + 64));
    v13 = *(_QWORD *)a2;
    v14 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
    if ( *(_QWORD *)a2 != v14 )
    {
      do
      {
        v15 = *(_QWORD *)(v13 + 8);
        v16 = *(_DWORD *)v13;
        v13 += 16;
        sub_B99FD0(v12, v16, v15);
      }
      while ( v14 != v13 );
    }
  }
  else
  {
    (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 8) + 928LL))(*(_QWORD *)(a1 + 8), *(_QWORD *)(a3 + 40));
    HIDWORD(v18) = 0;
    v22 = 257;
    return sub_B33D10(a2, 0x154u, 0, 0, 0, 0, (unsigned int)v18, (__int64)v21);
  }
  return v12;
}
