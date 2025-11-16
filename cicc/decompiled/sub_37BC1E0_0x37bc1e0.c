// Function: sub_37BC1E0
// Address: 0x37bc1e0
//
__int64 **__fastcall sub_37BC1E0(__int64 **a1, __int64 *a2, int *a3)
{
  __int64 v4; // rdx
  __int64 **result; // rax
  __int64 v6; // r10
  int v7; // ecx
  unsigned int v8; // edi
  int *v9; // r8
  int v10; // r9d
  __int64 *v11; // rcx
  int v12; // r8d
  __int64 *v13; // rdx
  int v14; // ebx

  v4 = *((unsigned int *)a2 + 6);
  result = a1;
  v6 = a2[1];
  if ( (_DWORD)v4 )
  {
    v7 = *a3;
    v8 = (v4 - 1) & (37 * v7);
    v9 = (int *)(v6 + 88LL * v8);
    v10 = *v9;
    if ( *v9 == v7 )
    {
LABEL_3:
      v11 = (__int64 *)*a2;
      *result = a2;
      result[2] = (__int64 *)v9;
      result[1] = v11;
      result[3] = (__int64 *)(v6 + 88 * v4);
      return result;
    }
    v12 = 1;
    while ( v10 != -1 )
    {
      v14 = v12 + 1;
      v8 = (v4 - 1) & (v12 + v8);
      v9 = (int *)(v6 + 88LL * v8);
      v10 = *v9;
      if ( v7 == *v9 )
        goto LABEL_3;
      v12 = v14;
    }
  }
  *result = a2;
  v13 = (__int64 *)(v6 + 88 * v4);
  result[1] = (__int64 *)*a2;
  result[2] = v13;
  result[3] = v13;
  return result;
}
