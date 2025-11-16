// Function: sub_16D9420
// Address: 0x16d9420
//
__int64 __fastcall sub_16D9420(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 (*i)(); // rsi
  __int64 v8; // r12
  __int64 v9; // rax
  _QWORD *v10; // r13
  _QWORD *v11; // r12
  _QWORD *v12; // rdi
  _QWORD *v13; // rdi
  _QWORD *v14; // rdi
  __int64 result; // rax

  for ( i = (__int64 (*)())a1[8]; i; i = (__int64 (*)())a1[8] )
    sub_16D9260(a1, i, a3, a4, a5, a6);
  if ( !qword_4FA1610 )
    sub_16C1EA0((__int64)&qword_4FA1610, sub_160CFB0, (__int64)sub_160D0B0, a4, a5, a6);
  v8 = qword_4FA1610;
  if ( (unsigned __int8)sub_16D5D40() )
    sub_16C30C0((pthread_mutex_t **)v8);
  else
    ++*(_DWORD *)(v8 + 8);
  v9 = a1[13];
  *(_QWORD *)a1[12] = v9;
  if ( v9 )
    *(_QWORD *)(v9 + 96) = a1[12];
  if ( (unsigned __int8)sub_16D5D40() )
    sub_16C30E0((pthread_mutex_t **)v8);
  else
    --*(_DWORD *)(v8 + 8);
  v10 = (_QWORD *)a1[10];
  v11 = (_QWORD *)a1[9];
  if ( v10 != v11 )
  {
    do
    {
      v12 = (_QWORD *)v11[8];
      if ( v12 != v11 + 10 )
        j_j___libc_free_0(v12, v11[10] + 1LL);
      v13 = (_QWORD *)v11[4];
      if ( v13 != v11 + 6 )
        j_j___libc_free_0(v13, v11[6] + 1LL);
      v11 += 12;
    }
    while ( v10 != v11 );
    v11 = (_QWORD *)a1[9];
  }
  if ( v11 )
    j_j___libc_free_0(v11, a1[11] - (_QWORD)v11);
  v14 = (_QWORD *)a1[4];
  if ( v14 != a1 + 6 )
    j_j___libc_free_0(v14, a1[6] + 1LL);
  result = (__int64)(a1 + 2);
  if ( (_QWORD *)*a1 != a1 + 2 )
    return j_j___libc_free_0(*a1, a1[2] + 1LL);
  return result;
}
