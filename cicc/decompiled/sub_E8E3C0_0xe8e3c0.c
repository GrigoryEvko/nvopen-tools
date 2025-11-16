// Function: sub_E8E3C0
// Address: 0xe8e3c0
//
__int64 __fastcall sub_E8E3C0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r14
  __int64 v9; // r8
  __int64 *v10; // r15
  __int64 v11; // rax
  __int64 (*v12)(); // r9
  __int64 (*v13)(); // rax
  __int64 result; // rax
  char v15; // al
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r9
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 (*v21)(); // rax
  void (*v22)(); // rcx
  char v23; // al
  __int64 v24; // [rsp+8h] [rbp-B8h]
  _QWORD v25[2]; // [rsp+10h] [rbp-B0h] BYREF
  _QWORD v26[2]; // [rsp+20h] [rbp-A0h] BYREF
  _BYTE v27[144]; // [rsp+30h] [rbp-90h] BYREF

  sub_E9A410();
  v4 = *(_QWORD *)(a1[36] + 8LL);
  *(_BYTE *)(v4 + 48) |= 2u;
  sub_E7BC40(a1, *(unsigned int **)(a1[36] + 8LL), v5, v6, v7);
  v8 = a1[37];
  v9 = a2;
  v10 = *(__int64 **)(v8 + 8);
  v11 = *v10;
  v12 = *(__int64 (**)())(*v10 + 120);
  if ( v12 != sub_E5B830 )
  {
    v15 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64))v12)(*(_QWORD *)(v8 + 8), a2, a3);
    v9 = a2;
    if ( v15 )
      goto LABEL_5;
    v11 = *v10;
  }
  v13 = *(__int64 (**)())(v11 + 24);
  if ( v13 == sub_E8A170 )
    return (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 1320LL))(a1, v9, a3);
  v24 = v9;
  v23 = ((__int64 (__fastcall *)(__int64 *))v13)(v10);
  v9 = v24;
  if ( !v23 )
    return (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 1320LL))(a1, v9, a3);
LABEL_5:
  if ( !*(_BYTE *)(v8 + 33) )
  {
    v17 = *(unsigned int *)(v8 + 368);
    if ( !(_DWORD)v17 )
      return (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 1336LL))(a1, v9, a3);
    v16 = *(unsigned int *)(v4 + 40);
    if ( !(_DWORD)v16 )
      return (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 1336LL))(a1, v9, a3);
  }
  v19 = *(_QWORD *)v9;
  v26[0] = v27;
  v25[0] = v19;
  v25[1] = *(_QWORD *)(v9 + 8);
  v26[1] = 0x600000000LL;
  if ( *(_DWORD *)(v9 + 24) )
    sub_E8A430((__int64)v26, v9 + 16, v16, v17, v9, v18);
LABEL_11:
  v20 = *v10;
  while ( 1 )
  {
    v21 = *(__int64 (**)())(v20 + 120);
    if ( v21 == sub_E5B830 || !((unsigned __int8 (__fastcall *)(__int64 *, _QWORD *, __int64))v21)(v10, v25, a3) )
      break;
    v20 = *v10;
    v22 = *(void (**)())(*v10 + 144);
    if ( v22 != nullsub_322 )
    {
      ((void (__fastcall *)(__int64 *, _QWORD *, __int64))v22)(v10, v25, a3);
      goto LABEL_11;
    }
  }
  result = (*(__int64 (__fastcall **)(_QWORD *, _QWORD *, __int64))(*a1 + 1320LL))(a1, v25, a3);
  if ( (_BYTE *)v26[0] != v27 )
    return _libc_free(v26[0], v25);
  return result;
}
