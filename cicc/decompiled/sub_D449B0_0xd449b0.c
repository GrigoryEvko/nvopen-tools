// Function: sub_D449B0
// Address: 0xd449b0
//
__int64 *__fastcall sub_D449B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // r14
  unsigned __int64 v5; // r13
  unsigned __int8 v6; // al
  __int64 v7; // rsi
  __int64 *v8; // rdi
  unsigned __int8 v9; // al
  __int64 *v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 *result; // rax
  __int64 v18; // rdx
  __int64 *v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rsi
  unsigned __int64 v23; // [rsp+8h] [rbp-88h]
  unsigned __int64 v24; // [rsp+10h] [rbp-80h]
  char v25; // [rsp+1Fh] [rbp-71h]
  __int64 v26; // [rsp+28h] [rbp-68h] BYREF
  unsigned __int64 v27[12]; // [rsp+30h] [rbp-60h] BYREF

  v3 = *(_QWORD *)a1;
  v4 = *(_QWORD *)(a1 + 48);
  v5 = *(_QWORD *)(a1 + 56);
  v24 = *(_QWORD *)(a1 + 32);
  v23 = *(_QWORD *)(a1 + 40);
  v25 = *(_BYTE *)(a1 + 64);
  v26 = *(_QWORD *)(a1 + 8);
  if ( !v4
    || ((v6 = *(_BYTE *)(v4 - 16), (v6 & 2) != 0)
      ? (__int64 *)(v8 = *(__int64 **)(v4 - 32), v7 = *(unsigned int *)(v4 - 24))
      : (v7 = (*(_WORD *)(v4 - 16) >> 6) & 0xF, v8 = (__int64 *)(v4 - 8LL * ((v6 >> 2) & 0xF) - 16)),
        &v8[v7] != sub_D338A0(v8, (__int64)&v8[v7], v3)) )
  {
    v4 = 0;
  }
  if ( !v5
    || ((v9 = *(_BYTE *)(v5 - 16), (v9 & 2) == 0)
      ? (v11 = (*(_WORD *)(v5 - 16) >> 6) & 0xF, v10 = (__int64 *)(v5 - 8LL * ((v9 >> 2) & 0xF) - 16))
      : (__int64 *)(v10 = *(__int64 **)(v5 - 32), v11 = *(unsigned int *)(v5 - 24)),
        &v10[v11] != sub_D338A0(v10, (__int64)&v10[v11], v3)) )
  {
    v5 = 0;
  }
  v27[5] = v5;
  v27[0] = a2;
  v27[2] = v24;
  v27[1] = -1;
  v27[3] = v23;
  v27[4] = v4;
  sub_FD9690(v3 + 976, v27);
  v27[0] = a2 & 0xFFFFFFFFFFFFFFFBLL;
  v12 = sub_D40250(v3, v27);
  result = (__int64 *)sub_D44380(v12, &v26, v13, v14, v15, v16);
  if ( !v25 )
    return result;
  if ( !*(_BYTE *)(v3 + 164) )
    return sub_C8CC70(v3 + 136, a2, v18, (__int64)v19, v20, v21);
  result = *(__int64 **)(v3 + 144);
  v22 = *(unsigned int *)(v3 + 156);
  v19 = &result[v22];
  if ( result == v19 )
  {
LABEL_17:
    if ( (unsigned int)v22 < *(_DWORD *)(v3 + 152) )
    {
      *(_DWORD *)(v3 + 156) = v22 + 1;
      *v19 = a2;
      ++*(_QWORD *)(v3 + 136);
      return result;
    }
    return sub_C8CC70(v3 + 136, a2, v18, (__int64)v19, v20, v21);
  }
  while ( a2 != *result )
  {
    if ( v19 == ++result )
      goto LABEL_17;
  }
  return result;
}
