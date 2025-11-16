// Function: sub_2E18770
// Address: 0x2e18770
//
__int64 *__fastcall sub_2E18770(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // r12
  __int64 v8; // r13
  __int64 *result; // rax
  __int64 v10; // rcx
  __int64 *v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rbx
  __int64 *v14; // rax
  char v15; // dl
  __m128i v16; // [rsp+0h] [rbp-40h] BYREF
  char v17; // [rsp+10h] [rbp-30h]

  v7 = a1[2];
  do
  {
    v8 = *(_QWORD *)(v7 - 24);
    if ( !*(_BYTE *)(v7 - 8) )
    {
      result = *(__int64 **)(v8 + 112);
      *(_BYTE *)(v7 - 8) = 1;
      *(_QWORD *)(v7 - 16) = result;
      goto LABEL_4;
    }
    while ( 1 )
    {
      result = *(__int64 **)(v7 - 16);
LABEL_4:
      v10 = *(unsigned int *)(v8 + 120);
      if ( result == (__int64 *)(*(_QWORD *)(v8 + 112) + 8 * v10) )
        break;
      v11 = result + 1;
      *(_QWORD *)(v7 - 16) = result + 1;
      v12 = *a1;
      v13 = *result;
      if ( !*(_BYTE *)(*a1 + 28) )
        goto LABEL_11;
      v14 = *(__int64 **)(v12 + 8);
      v10 = *(unsigned int *)(v12 + 20);
      v11 = &v14[v10];
      if ( v14 == v11 )
      {
LABEL_13:
        if ( (unsigned int)v10 < *(_DWORD *)(v12 + 16) )
        {
          *(_DWORD *)(v12 + 20) = v10 + 1;
          *v11 = v13;
          ++*(_QWORD *)v12;
LABEL_12:
          v16.m128i_i64[0] = v13;
          v17 = 0;
          return (__int64 *)sub_2E18730(a1 + 1, &v16);
        }
LABEL_11:
        sub_C8CC70(v12, v13, (__int64)v11, v10, a5, a6);
        if ( v15 )
          goto LABEL_12;
      }
      else
      {
        while ( v13 != *v14 )
        {
          if ( v11 == ++v14 )
            goto LABEL_13;
        }
      }
    }
    a1[2] -= 24LL;
    v7 = a1[2];
  }
  while ( v7 != a1[1] );
  return result;
}
