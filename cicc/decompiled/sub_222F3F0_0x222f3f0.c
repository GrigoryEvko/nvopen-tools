// Function: sub_222F3F0
// Address: 0x222f3f0
//
__int64 *__fastcall sub_222F3F0(__int64 *a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  const char *v6; // rbx
  unsigned __int64 v7; // r13
  _BYTE *v8; // rbp
  __int64 v9; // rcx
  __int64 v10; // r14
  __int64 v11; // rax
  _BYTE *v12; // rdx
  unsigned __int64 v13; // rax
  size_t v14; // rdx
  const char *v15; // rbx
  int v17; // edx
  const char *v18; // [rsp+0h] [rbp-58h]
  __int64 v19; // [rsp+8h] [rbp-50h]
  const char *v20; // [rsp+18h] [rbp-40h]

  v6 = (const char *)&unk_4FD67D8;
  *a1 = (__int64)&unk_4FD67D8;
  if ( a3 != (_BYTE *)a4 )
  {
    if ( !a3 )
      sub_426248((__int64)"basic_string::_S_construct null not valid");
    v6 = (const char *)sub_222EC60(a3, a4);
  }
  v20 = v6;
  v7 = 2 * (a4 - (_QWORD)a3);
  v18 = &v6[*((_QWORD *)v6 - 3)];
  v8 = (_BYTE *)sub_2207820(v7);
  while ( 1 )
  {
    v13 = sub_2255050(a2, v8, v6, v7);
    v14 = v13;
    if ( v7 <= v13 )
    {
      v7 = v13 + 1;
      j_j___libc_free_0_0((unsigned __int64)v8);
      v8 = (_BYTE *)sub_2207820(v7);
      v14 = sub_2255050(a2, v8, v6, v7);
    }
    sub_2215BF0(a1, v8, v14);
    v15 = &v6[strlen(v6)];
    if ( v18 == v15 )
      break;
    v6 = v15 + 1;
    v9 = *(_QWORD *)(*a1 - 24);
    v19 = v9;
    v10 = v9 + 1;
    if ( (unsigned __int64)(v9 + 1) > *(_QWORD *)(*a1 - 16) || *(int *)(*a1 - 8) > 0 )
      sub_2215AB0(a1, v9 + 1);
    *(_BYTE *)(*a1 + *(_QWORD *)(*a1 - 24)) = 0;
    v11 = *a1;
    v12 = (_BYTE *)(*a1 - 24);
    if ( v12 != (_BYTE *)&unk_4FD67C0 )
    {
      *(_DWORD *)(v11 - 8) = 0;
      *(_QWORD *)(v11 - 24) = v10;
      v12[v19 + 25] = 0;
    }
  }
  j_j___libc_free_0_0((unsigned __int64)v8);
  if ( v20 - 24 != (const char *)&unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v17 = _InterlockedExchangeAdd((volatile signed __int32 *)v20 - 2, 0xFFFFFFFF);
    }
    else
    {
      v17 = *((_DWORD *)v20 - 2);
      *((_DWORD *)v20 - 2) = v17 - 1;
    }
    if ( v17 <= 0 )
      j_j___libc_free_0_1((unsigned __int64)(v20 - 24));
  }
  return a1;
}
