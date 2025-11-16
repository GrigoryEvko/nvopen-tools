// Function: sub_39DFC90
// Address: 0x39dfc90
//
void __fastcall sub_39DFC90(_QWORD *a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // r13
  __int64 v6; // rdi
  unsigned __int64 v7; // r13
  __int64 (__fastcall *v8)(__int64); // rax
  __int64 *v9; // r14
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax

  v2 = (__int64)(a1 + 80);
  *(_QWORD *)(v2 - 640) = off_4A40718;
  sub_16E79F0(v2);
  a1[74] = &unk_49EFD28;
  sub_16E7960((__int64)(a1 + 74));
  v3 = a1[56];
  if ( (_QWORD *)v3 != a1 + 58 )
    _libc_free(v3);
  v4 = a1[38];
  if ( (_QWORD *)v4 != a1 + 40 )
    _libc_free(v4);
  v5 = a1[37];
  if ( v5 )
  {
    sub_390A9D0(a1[37]);
    j_j___libc_free_0(v5);
  }
  v6 = a1[36];
  if ( v6 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
  v7 = a1[33];
  if ( v7 )
  {
    v8 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL);
    if ( v8 == sub_16BE0F0 )
    {
      *(_QWORD *)v7 = &unk_49EF340;
      if ( *(_QWORD *)(v7 + 24) != *(_QWORD *)(v7 + 8) )
        sub_16E7BA0((__int64 *)v7);
      v9 = *(__int64 **)(v7 + 40);
      if ( v9 )
      {
        v10 = *(_QWORD *)(v7 + 8);
        if ( !*(_DWORD *)(v7 + 32) || v10 )
        {
          v12 = *(_QWORD *)(v7 + 16) - v10;
        }
        else
        {
          v11 = sub_16E7720();
          v9 = *(__int64 **)(v7 + 40);
          v12 = v11;
        }
        v13 = v9[3];
        v14 = v9[1];
        if ( v12 )
        {
          if ( v14 != v13 )
            sub_16E7BA0(v9);
          v15 = sub_2207820(v12);
          sub_16E7A40((__int64)v9, v15, v12, 1);
        }
        else
        {
          if ( v14 != v13 )
            sub_16E7BA0(v9);
          sub_16E7A40((__int64)v9, 0, 0, 0);
        }
      }
      sub_16E7960(v7);
      j_j___libc_free_0(v7);
    }
    else
    {
      v8(a1[33]);
    }
  }
  sub_38DCBC0(a1);
}
