// Function: sub_1E857B0
// Address: 0x1e857b0
//
__int64 __fastcall sub_1E857B0(__int64 a1, const char *a2, __int64 *a3)
{
  _QWORD *v5; // rdi
  _BYTE *v6; // rax
  int v7; // eax
  void *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r15
  void (__fastcall *v12)(__int64, void *, _QWORD); // r12
  void *v13; // rax
  void *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r12
  char *v19; // rax
  size_t v20; // rdx
  void *v21; // rdi
  __int64 v23; // r12
  void *v24; // rax
  size_t v25; // [rsp+8h] [rbp-28h]

  v5 = sub_16E8CB0();
  v6 = (_BYTE *)v5[3];
  if ( (unsigned __int64)v6 >= v5[2] )
  {
    sub_16E7DE0((__int64)v5, 10);
  }
  else
  {
    v5[3] = v6 + 1;
    *v6 = 10;
  }
  v7 = *(_DWORD *)(a1 + 56);
  *(_DWORD *)(a1 + 56) = v7 + 1;
  if ( !v7 )
  {
    if ( *(_QWORD *)(a1 + 8) )
    {
      v8 = sub_16E8CB0();
      v9 = sub_1263B40((__int64)v8, "# ");
      v10 = sub_1263B40(v9, *(const char **)(a1 + 8));
      sub_1549FC0(v10, 0xAu);
    }
    v11 = *(_QWORD *)(a1 + 568);
    if ( v11 )
    {
      v12 = *(void (__fastcall **)(__int64, void *, _QWORD))(*(_QWORD *)v11 + 40LL);
      v13 = sub_16E8CB0();
      v12(v11, v13, 0);
    }
    else
    {
      v23 = *(_QWORD *)(a1 + 584);
      v24 = sub_16E8CB0();
      sub_1E0B0B0((__int64)a3, (__int64)v24, v23);
    }
  }
  v14 = sub_16E8CB0();
  v15 = sub_1263B40((__int64)v14, "*** Bad machine code: ");
  v16 = sub_1263B40(v15, a2);
  v17 = sub_1263B40(v16, " ***\n");
  v18 = sub_1263B40(v17, "- function:    ");
  v19 = (char *)sub_1E0A440(a3);
  v21 = *(void **)(v18 + 24);
  if ( v20 > *(_QWORD *)(v18 + 16) - (_QWORD)v21 )
  {
    v18 = sub_16E7EE0(v18, v19, v20);
  }
  else if ( v20 )
  {
    v25 = v20;
    memcpy(v21, v19, v20);
    *(_QWORD *)(v18 + 24) += v25;
  }
  return sub_1263B40(v18, "\n");
}
