// Function: sub_2FC1390
// Address: 0x2fc1390
//
__int64 *__fastcall sub_2FC1390(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 *result; // rax
  __int64 v9; // rcx
  __int64 *v10; // rdx
  __int64 v11; // rbx
  __int64 *v12; // rax
  char v13; // dl
  __m128i v14; // [rsp+0h] [rbp-40h] BYREF
  char v15; // [rsp+10h] [rbp-30h]

  v6 = *(_QWORD *)(a1 + 104);
  do
  {
    v7 = *(_QWORD *)(v6 - 24);
    if ( !*(_BYTE *)(v6 - 8) )
    {
      result = *(__int64 **)(v7 + 112);
      *(_BYTE *)(v6 - 8) = 1;
      *(_QWORD *)(v6 - 16) = result;
      goto LABEL_4;
    }
    while ( 1 )
    {
      result = *(__int64 **)(v6 - 16);
LABEL_4:
      v9 = *(unsigned int *)(v7 + 120);
      if ( result == (__int64 *)(*(_QWORD *)(v7 + 112) + 8 * v9) )
        break;
      v10 = result + 1;
      *(_QWORD *)(v6 - 16) = result + 1;
      v11 = *result;
      if ( !*(_BYTE *)(a1 + 28) )
        goto LABEL_11;
      v12 = *(__int64 **)(a1 + 8);
      v9 = *(unsigned int *)(a1 + 20);
      v10 = &v12[v9];
      if ( v12 == v10 )
      {
LABEL_13:
        if ( (unsigned int)v9 < *(_DWORD *)(a1 + 16) )
        {
          *(_DWORD *)(a1 + 20) = v9 + 1;
          *v10 = v11;
          ++*(_QWORD *)a1;
LABEL_12:
          v14.m128i_i64[0] = v11;
          v15 = 0;
          return (__int64 *)sub_2FC1350((unsigned __int64 *)(a1 + 96), &v14);
        }
LABEL_11:
        sub_C8CC70(a1, v11, (__int64)v10, v9, a5, a6);
        if ( v13 )
          goto LABEL_12;
      }
      else
      {
        while ( v11 != *v12 )
        {
          if ( v10 == ++v12 )
            goto LABEL_13;
        }
      }
    }
    *(_QWORD *)(a1 + 104) -= 24LL;
    v6 = *(_QWORD *)(a1 + 104);
  }
  while ( v6 != *(_QWORD *)(a1 + 96) );
  return result;
}
