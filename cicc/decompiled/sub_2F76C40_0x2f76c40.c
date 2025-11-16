// Function: sub_2F76C40
// Address: 0x2f76c40
//
__int64 __fastcall sub_2F76C40(_QWORD *a1, unsigned int a2, __int64 *a3, _QWORD *a4, __int64 a5, __int64 a6)
{
  char v6; // r14
  unsigned __int16 *v8; // rbx
  __int64 v9; // r15
  __int64 v10; // r13
  __int64 v11; // r13
  __int64 result; // rax
  __int64 v13; // r14
  __int128 v14; // [rsp-18h] [rbp-58h]
  __int128 v15; // [rsp-18h] [rbp-58h]
  char v16; // [rsp+4h] [rbp-3Ch]

  v6 = a5;
  v8 = (unsigned __int16 *)(*a1 + ((unsigned __int64)a2 << 6));
  v16 = a5;
  v9 = a3[26];
  v10 = v9 + 24LL * *((unsigned int *)a3 + 54);
  while ( v10 != v9 )
  {
    v14 = *(_OWORD *)(v9 + 8);
    v9 += 24;
    sub_2F76A30(v8, 1, a4, v6, a5, a6, *(_QWORD *)(v9 - 24), v14);
  }
  v11 = *a3;
  result = 3LL * *((unsigned int *)a3 + 2);
  v13 = *a3 + 24LL * *((unsigned int *)a3 + 2);
  if ( v13 != *a3 )
  {
    do
    {
      v15 = *(_OWORD *)(v11 + 8);
      v11 += 24;
      result = sub_2F76A30(v8, 0, a4, v16, a5, a6, *(_QWORD *)(v11 - 24), v15);
    }
    while ( v13 != v11 );
  }
  return result;
}
