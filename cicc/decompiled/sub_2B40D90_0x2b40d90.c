// Function: sub_2B40D90
// Address: 0x2b40d90
//
__int64 **__fastcall sub_2B40D90(__int64 **a1, __int64 *a2, __int64 a3)
{
  __int64 **result; // rax
  char v4; // bl
  __int64 *v5; // r11
  int v6; // r12d
  unsigned int v7; // ecx
  __int64 *v8; // r9
  __int64 v9; // rdi
  __int64 v10; // r10
  __int64 *v11; // rdx
  __int64 v12; // r8
  int i; // edi
  __int64 v14; // rdi
  __int64 *v15; // rdx
  __int64 *v16; // rdi

  result = a1;
  v4 = a2[1] & 1;
  if ( v4 )
  {
    v5 = a2 + 2;
    v6 = 15;
    v7 = ((unsigned __int8)((unsigned int)a3 >> 9) ^ (unsigned __int8)((unsigned int)a3 >> 4)) & 0xF;
    v8 = &a2[9 * (((unsigned __int8)((unsigned int)a3 >> 9) ^ (unsigned __int8)((unsigned int)a3 >> 4)) & 0xF) + 2];
    v9 = 144;
    v10 = *v8;
    if ( a3 == *v8 )
    {
LABEL_3:
      v11 = (__int64 *)*a2;
      *result = a2;
      result[2] = v8;
      result[1] = v11;
      result[3] = &v5[v9];
      return result;
    }
    goto LABEL_7;
  }
  v12 = *((unsigned int *)a2 + 6);
  v5 = (__int64 *)a2[2];
  if ( (_DWORD)v12 )
  {
    v6 = v12 - 1;
    v7 = (v12 - 1) & (((unsigned int)a3 >> 4) ^ ((unsigned int)a3 >> 9));
    v8 = &v5[9 * v7];
    v10 = *v8;
    if ( *v8 == a3 )
    {
LABEL_6:
      v9 = 9 * v12;
      goto LABEL_3;
    }
LABEL_7:
    for ( i = 1; ; ++i )
    {
      if ( v10 == -4096 )
      {
        if ( v4 )
        {
          v14 = 144;
          goto LABEL_11;
        }
        v12 = *((unsigned int *)a2 + 6);
        goto LABEL_13;
      }
      v7 = v6 & (i + v7);
      v8 = &v5[9 * v7];
      v10 = *v8;
      if ( *v8 == a3 )
        break;
    }
    if ( !v4 )
    {
      v12 = *((unsigned int *)a2 + 6);
      goto LABEL_6;
    }
    v9 = 144;
    goto LABEL_3;
  }
LABEL_13:
  v14 = 9 * v12;
LABEL_11:
  v15 = (__int64 *)*a2;
  v16 = &v5[v14];
  *result = a2;
  result[2] = v16;
  result[1] = v15;
  result[3] = v16;
  return result;
}
