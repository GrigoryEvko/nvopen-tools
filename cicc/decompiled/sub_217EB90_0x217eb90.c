// Function: sub_217EB90
// Address: 0x217eb90
//
__int64 **__fastcall sub_217EB90(__int64 **a1, __int64 *a2, int *a3)
{
  __int64 v4; // rdx
  __int64 v5; // r8
  __int64 **result; // rax
  int v7; // edi
  unsigned int v8; // r9d
  int *v9; // rcx
  int v10; // r10d
  __int64 *v11; // rdi
  int v12; // ecx
  __int64 *v13; // rdi
  __int64 *v14; // rdx
  int v15; // ebx
  __int64 *v16; // rcx
  int v17; // r10d
  __int64 *v18; // rdi
  __int64 *v19; // rdx

  v4 = *((unsigned int *)a2 + 6);
  v5 = a2[1];
  result = a1;
  if ( !(_DWORD)v4 )
    goto LABEL_5;
  v7 = *a3;
  v8 = (v4 - 1) & (37 * *a3);
  v9 = (int *)(v5 + 4LL * v8);
  v10 = *v9;
  if ( v7 == *v9 )
  {
    v11 = (__int64 *)*a2;
    *result = a2;
    result[2] = (__int64 *)v9;
    result[1] = v11;
    result[3] = (__int64 *)(v5 + 4 * v4);
    return result;
  }
  v12 = 1;
  if ( v10 != -1 )
  {
    do
    {
      v15 = v12 + 1;
      v8 = (v4 - 1) & (v12 + v8);
      v16 = (__int64 *)(v5 + 4LL * v8);
      v17 = *(_DWORD *)v16;
      if ( v7 == *(_DWORD *)v16 )
      {
        v18 = (__int64 *)*a2;
        v19 = (__int64 *)(v5 + 4 * v4);
        goto LABEL_8;
      }
      v12 = v15;
    }
    while ( v17 != -1 );
    v16 = (__int64 *)(v5 + 4 * v4);
    v18 = (__int64 *)*a2;
    v19 = v16;
LABEL_8:
    *result = a2;
    result[1] = v18;
    result[2] = v16;
    result[3] = v19;
  }
  else
  {
LABEL_5:
    v13 = (__int64 *)*a2;
    *result = a2;
    v14 = (__int64 *)(v5 + 4 * v4);
    result[2] = v14;
    result[1] = v13;
    result[3] = v14;
  }
  return result;
}
