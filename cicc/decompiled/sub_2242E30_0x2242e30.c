// Function: sub_2242E30
// Address: 0x2242e30
//
__int64 *__fastcall sub_2242E30(__int64 *a1, __int64 a2, wchar_t *a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 *v5; // r14
  const wchar_t *v8; // rbx
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // r12
  unsigned __int64 v11; // rbp
  __int64 v12; // rcx
  __int64 v13; // r13
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  size_t v16; // rdx
  const wchar_t *v17; // rbx
  int v19; // edx
  const wchar_t *v20; // [rsp+8h] [rbp-50h]
  const wchar_t *v21; // [rsp+18h] [rbp-40h]

  v4 = a2;
  v5 = a1;
  v8 = (const wchar_t *)&unk_4FD67F8;
  *a1 = (__int64)&unk_4FD67F8;
  if ( a3 != (wchar_t *)a4 )
  {
    if ( !a3 )
      sub_426248((__int64)"basic_string::_S_construct null not valid");
    a2 = a4;
    a1 = (__int64 *)a3;
    v8 = (const wchar_t *)sub_2242590(a3, a4);
  }
  v21 = v8;
  v9 = a4 - (_QWORD)a3;
  v20 = &v8[*((_QWORD *)v8 - 3)];
  v10 = (a4 - (__int64)a3) >> 1;
  if ( v9 > 0x3FFFFFFFFFFFFFFCLL )
    sub_426640(a1, a2);
  v11 = sub_2207820(4 * v10);
  while ( 1 )
  {
    v15 = sub_22550A0(v4, v11, v8, v10);
    v16 = v15;
    if ( v10 <= v15 )
    {
      v10 = v15 + 1;
      j_j___libc_free_0_0(v11);
      if ( v10 > 0x1FFFFFFFFFFFFFFELL )
        sub_426640(v11, v11);
      v11 = sub_2207820(4 * v10);
      v16 = sub_22550A0(v4, v11, v8, v10);
    }
    sub_2216880(v5, v11, v16);
    v17 = &v8[wcslen(v8)];
    if ( v20 == v17 )
      break;
    v8 = v17 + 1;
    v12 = *(_QWORD *)(*v5 - 24);
    v13 = v12 + 1;
    if ( (unsigned __int64)(v12 + 1) > *(_QWORD *)(*v5 - 16) || *(int *)(*v5 - 8) > 0 )
      sub_2216730(v5, v12 + 1);
    v14 = *v5;
    *(_DWORD *)(v14 + 4LL * *(_QWORD *)(*v5 - 24)) = 0;
    if ( (_UNKNOWN *)(v14 - 24) != &unk_4FD67E0 )
    {
      *(_DWORD *)(v14 - 8) = 0;
      *(_QWORD *)(v14 - 24) = v13;
      *(_DWORD *)(v14 + 4 * v13) = 0;
    }
  }
  j_j___libc_free_0_0(v11);
  if ( v21 - 6 != (const wchar_t *)&unk_4FD67E0 )
  {
    if ( &_pthread_key_create )
    {
      v19 = _InterlockedExchangeAdd((volatile signed __int32 *)v21 - 2, 0xFFFFFFFF);
    }
    else
    {
      v19 = *(v21 - 2);
      *((_DWORD *)v21 - 2) = v19 - 1;
    }
    if ( v19 <= 0 )
      j_j___libc_free_0_2((unsigned __int64)(v21 - 6));
  }
  return v5;
}
