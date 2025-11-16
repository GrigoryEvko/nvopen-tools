// Function: sub_3112020
// Address: 0x3112020
//
__int64 *__fastcall sub_3112020(__int64 *a1, int **a2, __int64 *a3)
{
  int *v5; // r14
  __int64 v6; // r13
  __int64 *(__fastcall *v7)(__int64 *, __int64); // rax
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned __int8 *v10; // rdi
  size_t v11; // rsi
  int *v13; // rax
  char v14[8]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v15[3]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v16; // [rsp+40h] [rbp-30h]

  if ( (*(unsigned __int8 (__fastcall **)(int *, void *))(*(_QWORD *)*a2 + 48LL))(*a2, &unk_5031F50) )
  {
    v5 = *a2;
    *a2 = 0;
    v6 = *a3;
    v7 = *(__int64 *(__fastcall **)(__int64 *, __int64))(*(_QWORD *)v5 + 24LL);
    if ( v7 == sub_3111B20 )
      sub_3111610((__int64 *)v14, v5[2], (__int64)(v5 + 4));
    else
      v7((__int64 *)v14, (__int64)v5);
    v10 = *(unsigned __int8 **)v6;
    v11 = *(_QWORD *)(v6 + 8);
    LOWORD(v16) = 260;
    v15[2] = v14;
    sub_3111E70(v10, v11, (unsigned __int8 *)byte_3F871B3, 0, v8, v9, (char)v14);
    if ( *(_QWORD **)v14 != v15 )
    {
      v11 = v15[0] + 1LL;
      j_j___libc_free_0(*(unsigned __int64 *)v14);
    }
    *a1 = 1;
    (*(void (__fastcall **)(int *, size_t))(*(_QWORD *)v5 + 8LL))(v5, v11);
    return a1;
  }
  else
  {
    v13 = *a2;
    *a2 = 0;
    *a1 = (unsigned __int64)v13 | 1;
    return a1;
  }
}
