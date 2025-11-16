// Function: sub_EDD130
// Address: 0xedd130
//
__int64 __fastcall sub_EDD130(__int64 a1, unsigned int **a2, unsigned __int64 a3)
{
  unsigned int *v5; // rsi
  char v6; // al
  unsigned __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rsi
  char v10; // al
  unsigned int *v11; // rdi
  __int64 v13; // [rsp+8h] [rbp-38h] BYREF
  unsigned int *v14; // [rsp+10h] [rbp-30h] BYREF
  char v15; // [rsp+18h] [rbp-28h]

  v5 = *a2;
  sub_ED2120((__int64)&v14, v5, a3, *(_DWORD *)(a1 + 32));
  v6 = v15;
  v15 &= ~2u;
  if ( (v6 & 1) != 0 )
  {
    v7 = (unsigned __int64)v14;
    v14 = 0;
    v13 = v7 | 1;
    if ( (v7 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v13, (__int64)v5);
    v8 = 0;
  }
  else
  {
    v8 = (__int64)v14;
  }
  v9 = *(_QWORD *)(a1 + 8) - 80LL;
  sub_ED5670(v8, v9, 0);
  v10 = v15;
  if ( (v15 & 2) != 0 )
    sub_EDD0C0(&v14, v9);
  v11 = v14;
  *a2 = (unsigned int *)((char *)*a2 + *v14);
  if ( (v10 & 1) != 0 )
    (*(void (__fastcall **)(unsigned int *))(*(_QWORD *)v11 + 8LL))(v11);
  else
    j___libc_free_0(v11);
  return 1;
}
