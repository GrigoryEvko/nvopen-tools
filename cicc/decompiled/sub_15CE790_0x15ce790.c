// Function: sub_15CE790
// Address: 0x15ce790
//
__int64 **__fastcall sub_15CE790(__int64 **a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 **result; // rax
  __int64 v6; // r8
  unsigned int v7; // ecx
  __int64 *v8; // rdi
  __int64 v9; // r9
  __int64 *v10; // rcx
  int v11; // edi
  __int64 *v12; // rdx
  __int64 *v13; // rcx
  int v14; // ebx

  v4 = *((unsigned int *)a2 + 6);
  result = a1;
  v6 = a2[1];
  if ( (_DWORD)v4 )
  {
    v7 = (v4 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v8 = (__int64 *)(v6 + 56LL * v7);
    v9 = *v8;
    if ( *v8 == a3 )
    {
LABEL_3:
      v10 = (__int64 *)*a2;
      *result = a2;
      result[2] = v8;
      result[1] = v10;
      result[3] = (__int64 *)(v6 + 56 * v4);
      return result;
    }
    v11 = 1;
    while ( v9 != -8 )
    {
      v14 = v11 + 1;
      v7 = (v4 - 1) & (v11 + v7);
      v8 = (__int64 *)(v6 + 56LL * v7);
      v9 = *v8;
      if ( *v8 == a3 )
        goto LABEL_3;
      v11 = v14;
    }
  }
  *result = a2;
  v12 = (__int64 *)(v6 + 56 * v4);
  v13 = (__int64 *)*a2;
  result[2] = v12;
  result[1] = v13;
  result[3] = v12;
  return result;
}
