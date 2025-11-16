// Function: sub_F672C0
// Address: 0xf672c0
//
__int64 *__fastcall sub_F672C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v5; // r13
  bool v6; // zf
  unsigned __int64 v7; // rax
  __int64 *v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // r12
  __int64 *result; // rax
  __int64 v13; // rbx
  __int64 v14; // rsi
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  __int64 v18[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = *(__int64 **)a2;
  v7 = *(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  v6 = v7 == 0;
  v8 = *(__int64 **)a2;
  v9 = v7 - 24;
  v10 = 0;
  if ( !v6 )
    v10 = v9;
  v18[0] = v10;
  v11 = &v5[*(unsigned int *)(a2 + 8)];
  result = sub_F66DF0(v8, (__int64)v11, v18);
  if ( v11 != result )
    return result;
  if ( v5 == v11 )
  {
LABEL_17:
    v13 = *v11;
    return (__int64 *)sub_AA4AF0(a1, v13);
  }
  while ( 1 )
  {
    v13 = *v5;
    v14 = *(_QWORD *)(*v5 + 32);
    if ( v14 != *(_QWORD *)(a1 + 72) + 72LL )
    {
      if ( v14 )
        v14 -= 24;
      if ( !*(_BYTE *)(a3 + 84) )
      {
        if ( sub_C8CA60(a3 + 56, v14) )
          return (__int64 *)sub_AA4AF0(a1, v13);
        goto LABEL_15;
      }
      v15 = *(_QWORD **)(a3 + 64);
      v16 = &v15[*(unsigned int *)(a3 + 76)];
      if ( v15 != v16 )
        break;
    }
LABEL_15:
    if ( v11 == ++v5 )
    {
      v11 = *(__int64 **)a2;
      goto LABEL_17;
    }
  }
  while ( v14 != *v15 )
  {
    if ( v16 == ++v15 )
      goto LABEL_15;
  }
  return (__int64 *)sub_AA4AF0(a1, v13);
}
