// Function: sub_E7EC80
// Address: 0xe7ec80
//
__int64 __fastcall sub_E7EC80(_QWORD *a1, __int64 a2, __int64 a3, size_t *a4, int a5, unsigned __int64 *a6, __int64 a7)
{
  __int64 v9; // rbx
  __int64 result; // rax
  int *v11; // rbx
  __int64 v12; // r14
  __int64 v13; // rsi
  int v14; // eax
  __int64 v15; // rsi
  int *v16; // r12
  __int64 v17; // rbx
  __int64 v18; // rdi
  __int64 v19; // rdi
  unsigned __int64 v21; // rax
  char v22[32]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v23; // [rsp+20h] [rbp-40h]

  if ( *a6 )
  {
    (*(void (__fastcall **)(_QWORD *, unsigned __int64, _QWORD))(*a1 + 176LL))(a1, *a6, 0);
  }
  else
  {
    v19 = a1[1];
    v23 = 257;
    v21 = sub_E71CB0(v19, a4, a5, 0, 0, (__int64)v22, 0, -1, 0);
    *a6 = v21;
    (*(void (__fastcall **)(_QWORD *, unsigned __int64, _QWORD))(*a1 + 176LL))(a1, v21, 0);
    (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 536LL))(a1, 65, 1);
  }
  v9 = sub_E7EBC0((__int64)a1, (int **)a7);
  (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 536LL))(a1, v9 + a3 + 10, 4);
  (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 512LL))(a1, a2, a3);
  (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a1 + 536LL))(a1, 0, 1);
  (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 536LL))(a1, 1, 1);
  (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 536LL))(a1, v9 + 5, 4);
  result = *(unsigned int *)(a7 + 8);
  v11 = *(int **)a7;
  v12 = *(_QWORD *)a7 + 48 * result;
  if ( v12 == *(_QWORD *)a7 )
    goto LABEL_15;
  do
  {
    while ( 1 )
    {
      sub_E98EB0(a1, (unsigned int)v11[1], 0);
      v14 = *v11;
      if ( *v11 == 2 )
        goto LABEL_6;
      if ( v14 != 3 )
        break;
      sub_E98EB0(a1, (unsigned int)v11[2], 0);
LABEL_6:
      v13 = *((_QWORD *)v11 + 2);
      v11 += 12;
      (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a1 + 512LL))(a1, v13, *((_QWORD *)v11 - 3));
      (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a1 + 536LL))(a1, 0, 1);
      if ( (int *)v12 == v11 )
        goto LABEL_11;
    }
    if ( v14 != 1 )
      BUG();
    v15 = (unsigned int)v11[2];
    v11 += 12;
    sub_E98EB0(a1, v15, 0);
  }
  while ( (int *)v12 != v11 );
LABEL_11:
  result = *(unsigned int *)(a7 + 8);
  v16 = *(int **)a7;
  v17 = *(_QWORD *)a7 + 48 * result;
  while ( v16 != (int *)v17 )
  {
    while ( 1 )
    {
      v17 -= 48;
      v18 = *(_QWORD *)(v17 + 16);
      result = v17 + 32;
      if ( v18 == v17 + 32 )
        break;
      result = j_j___libc_free_0(v18, *(_QWORD *)(v17 + 32) + 1LL);
      if ( v16 == (int *)v17 )
        goto LABEL_15;
    }
  }
LABEL_15:
  *(_DWORD *)(a7 + 8) = 0;
  return result;
}
