// Function: sub_393B230
// Address: 0x393b230
//
__int64 __fastcall sub_393B230(__int64 a1, unsigned int **a2, unsigned __int64 a3)
{
  unsigned int *v5; // rsi
  char v6; // al
  unsigned __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // rdx
  char v11; // al
  unsigned __int64 v12; // rdi
  __int64 v14; // [rsp+8h] [rbp-38h] BYREF
  unsigned int *v15; // [rsp+10h] [rbp-30h] BYREF
  char v16; // [rsp+18h] [rbp-28h]

  v5 = *a2;
  sub_1694E90((__int64)&v15, v5, a3, *(_DWORD *)(a1 + 32));
  v6 = v16;
  v16 &= ~2u;
  if ( (v6 & 1) != 0 )
  {
    v7 = (unsigned __int64)v15;
    v15 = 0;
    v14 = v7 | 1;
    if ( (v7 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_16BCAE0(&v14, (__int64)v5, v7 | 1);
    v8 = 0;
  }
  else
  {
    v8 = (__int64)v15;
  }
  v9 = *(_QWORD *)(a1 + 8) - 56LL;
  sub_16981C0(v8, v9, 0);
  v11 = v16;
  if ( (v16 & 2) != 0 )
    sub_393B1C0(&v15, v9, v10);
  v12 = (unsigned __int64)v15;
  *a2 = (unsigned int *)((char *)*a2 + *v15);
  if ( (v11 & 1) != 0 )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v12 + 8LL))(v12);
  else
    j___libc_free_0(v12);
  return 1;
}
