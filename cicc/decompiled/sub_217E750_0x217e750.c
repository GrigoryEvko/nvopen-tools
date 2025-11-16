// Function: sub_217E750
// Address: 0x217e750
//
__int64 **__fastcall sub_217E750(__int64 **a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // r9
  __int64 **result; // rax
  unsigned int v7; // ecx
  __int64 *v8; // rdi
  __int64 v9; // r8
  __int64 *v10; // rcx
  int v11; // edi
  __int64 *v12; // rcx
  __int64 *v13; // rdx
  int v14; // ebx
  __int64 *v15; // rdi
  __int64 v16; // r8
  __int64 *v17; // rcx
  __int64 *v18; // rdx

  v4 = *((unsigned int *)a2 + 6);
  v5 = a2[1];
  result = a1;
  if ( !(_DWORD)v4 )
    goto LABEL_5;
  v7 = (v4 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v8 = (__int64 *)(v5 + 8LL * v7);
  v9 = *v8;
  if ( a3 == *v8 )
  {
    v10 = (__int64 *)*a2;
    *result = a2;
    result[2] = v8;
    result[1] = v10;
    result[3] = (__int64 *)(v5 + 8 * v4);
    return result;
  }
  v11 = 1;
  if ( v9 != -8 )
  {
    do
    {
      v14 = v11 + 1;
      v7 = (v4 - 1) & (v11 + v7);
      v15 = (__int64 *)(v5 + 8LL * v7);
      v16 = *v15;
      if ( a3 == *v15 )
      {
        v17 = (__int64 *)*a2;
        v18 = (__int64 *)(v5 + 8 * v4);
        goto LABEL_8;
      }
      v11 = v14;
    }
    while ( v16 != -8 );
    v15 = (__int64 *)(v5 + 8 * v4);
    v17 = (__int64 *)*a2;
    v18 = v15;
LABEL_8:
    *result = a2;
    result[1] = v17;
    result[2] = v15;
    result[3] = v18;
  }
  else
  {
LABEL_5:
    v12 = (__int64 *)*a2;
    *result = a2;
    v13 = (__int64 *)(v5 + 8 * v4);
    result[2] = v13;
    result[1] = v12;
    result[3] = v13;
  }
  return result;
}
