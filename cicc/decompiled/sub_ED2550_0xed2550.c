// Function: sub_ED2550
// Address: 0xed2550
//
__int64 **__fastcall sub_ED2550(__int64 **a1, __int64 a2, __int64 a3, unsigned int a4, unsigned int a5, int a6)
{
  __int64 v8; // rax
  __int64 **result; // rax
  __int64 *v10; // rsi
  __int64 *v11; // rdx
  __int64 *v12; // r13
  __int64 v13; // r12
  unsigned __int64 i; // r8
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __int64 v17; // [rsp-10h] [rbp-30h]

  v8 = *(_QWORD *)(a3 + 48);
  if ( v8 )
    v8 = *(_QWORD *)(v8 + 24LL * a4);
  result = (__int64 **)(v8 + 24LL * a5);
  v10 = result[1];
  v11 = *result;
  v12 = *result;
  v13 = ((char *)v10 - (char *)*result) >> 4;
  if ( v13 )
  {
    if ( v11 == v10 )
    {
      i = 0;
    }
    else
    {
      for ( i = v11[1]; ; i = v16 )
      {
        v11 += 2;
        if ( v11 == v10 )
          break;
        v15 = v11[1];
        v16 = v15 + i;
        if ( v15 < i )
          v15 = i;
        if ( v16 < v15 )
          v16 = -1;
      }
    }
    sub_ED2230(a1, a2, v12, v13, i, a4, a6);
    return (__int64 **)v17;
  }
  return result;
}
