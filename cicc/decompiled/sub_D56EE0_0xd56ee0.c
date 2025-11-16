// Function: sub_D56EE0
// Address: 0xd56ee0
//
__int64 __fastcall sub_D56EE0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  char *v7; // rsi
  __int64 v8; // rdi
  _BYTE *v9; // rax
  int *v10; // r13
  int *v11; // rdi
  int *v13; // [rsp+8h] [rbp-28h] BYREF

  sub_D56E90((__int64 *)&v13, a3, *(_QWORD *)(a5 + 32));
  v7 = (char *)v13;
  if ( v13 )
  {
    v8 = sub_D52C50(*a2, v13);
    v9 = *(_BYTE **)(v8 + 32);
    if ( *(_BYTE **)(v8 + 24) == v9 )
    {
      v7 = "\n";
      sub_CB6200(v8, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v9 = 10;
      ++*(_QWORD *)(v8 + 32);
    }
    v10 = v13;
    if ( v13 )
    {
      v11 = (int *)*((_QWORD *)v13 + 1);
      if ( v11 != v13 + 6 )
        _libc_free(v11, v7);
      j_j___libc_free_0(v10, 88);
    }
  }
  *(_BYTE *)(a1 + 76) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = &unk_4F82400;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
