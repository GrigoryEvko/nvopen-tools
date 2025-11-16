// Function: sub_359B060
// Address: 0x359b060
//
__int64 **__fastcall sub_359B060(__int64 **a1, __int64 *a2, int *a3)
{
  __int64 v4; // rdx
  __int64 **result; // rax
  __int64 v6; // r8
  int v7; // ecx
  unsigned int v8; // edi
  int *v9; // r9
  int v10; // r10d
  __int64 *v11; // rcx
  int v12; // r9d
  __int64 *v13; // rcx
  __int64 *v14; // rdx
  int v15; // ebx

  v4 = *((unsigned int *)a2 + 6);
  result = a1;
  v6 = a2[1];
  if ( (_DWORD)v4 )
  {
    v7 = *a3;
    v8 = (v4 - 1) & (37 * v7);
    v9 = (int *)(v6 + 8LL * v8);
    v10 = *v9;
    if ( *v9 == v7 )
    {
LABEL_3:
      v11 = (__int64 *)*a2;
      *result = a2;
      result[2] = (__int64 *)v9;
      result[1] = v11;
      result[3] = (__int64 *)(v6 + 8 * v4);
      return result;
    }
    v12 = 1;
    while ( v10 != -1 )
    {
      v15 = v12 + 1;
      v8 = (v4 - 1) & (v12 + v8);
      v9 = (int *)(v6 + 8LL * v8);
      v10 = *v9;
      if ( v7 == *v9 )
        goto LABEL_3;
      v12 = v15;
    }
  }
  v13 = (__int64 *)*a2;
  v14 = (__int64 *)(v6 + 8 * v4);
  *result = a2;
  result[2] = v14;
  result[1] = v13;
  result[3] = v14;
  return result;
}
