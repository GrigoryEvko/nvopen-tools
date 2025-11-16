// Function: sub_1DA7E30
// Address: 0x1da7e30
//
__int64 __fastcall sub_1DA7E30(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 (*v4)(void); // rdx
  __int64 v5; // rax
  __int64 (*v6)(void); // rdx
  __int64 v7; // rax
  __int64 (*v8)(); // rax
  __int64 v9; // rax
  __int64 v10; // r15
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  _QWORD *v13; // r13
  unsigned __int64 v14; // rdi
  void (__fastcall *v16)(__int64, __int64, _QWORD *, _QWORD *); // [rsp+8h] [rbp-38h]

  if ( !sub_1626D20(*(_QWORD *)a2) )
    return 0;
  v3 = sub_1626D20(*(_QWORD *)a2);
  if ( !*(_DWORD *)(*(_QWORD *)(v3 + 8 * (5LL - *(unsigned int *)(v3 + 8))) + 36LL) )
    return 0;
  v4 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 112LL);
  v5 = 0;
  if ( v4 != sub_1D00B10 )
    v5 = v4();
  a1[29] = v5;
  v6 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 40LL);
  v7 = 0;
  if ( v6 != sub_1D00B00 )
    v7 = v6();
  a1[30] = v7;
  v8 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 48LL);
  if ( v8 == sub_1D90020 )
  {
    a1[31] = 0;
    BUG();
  }
  v9 = v8();
  a1[31] = v9;
  v10 = v9;
  v16 = *(void (__fastcall **)(__int64, __int64, _QWORD *, _QWORD *))(*(_QWORD *)v9 + 192LL);
  v11 = (_QWORD *)sub_22077B0(200);
  v12 = a1 + 32;
  v13 = v11;
  if ( v11 )
  {
    memset(v11, 0, 0xC8u);
    v11[6] = v11 + 8;
    v11[7] = 0x200000000LL;
    v16(v10, a2, v12, v11);
    _libc_free(v13[22]);
    _libc_free(v13[19]);
    _libc_free(v13[16]);
    _libc_free(v13[13]);
    v14 = v13[6];
    if ( v13 + 8 != (_QWORD *)v14 )
      _libc_free(v14);
    j_j___libc_free_0(v13, 200);
  }
  else
  {
    v16(v10, a2, v12, 0);
  }
  sub_20FC0D0(a1 + 35, a2);
  return sub_1DA4F80(a1, a2);
}
