// Function: sub_227B950
// Address: 0x227b950
//
__int64 **__fastcall sub_227B950(__int64 **a1, __int64 *a2, __int64 a3)
{
  char v5; // r9
  __int64 *v6; // r10
  int v7; // ebx
  __int64 v8; // rax
  unsigned int v9; // edi
  __int64 *v10; // rdx
  __int64 v11; // r11
  __int64 *v12; // rcx
  __int64 v14; // rax
  int i; // eax
  __int64 v16; // rax
  __int64 *v17; // r10
  __int64 *v18; // rax

  v5 = a2[1] & 1;
  if ( v5 )
  {
    v6 = a2 + 2;
    v7 = 7;
    v8 = 16;
    v9 = ((unsigned __int8)((unsigned int)a3 >> 9) ^ (unsigned __int8)((unsigned int)a3 >> 4)) & 7;
    v10 = &a2[2 * (((unsigned __int8)((unsigned int)a3 >> 9) ^ (unsigned __int8)((unsigned int)a3 >> 4)) & 7) + 2];
    v11 = *v10;
    if ( a3 == *v10 )
    {
LABEL_3:
      v12 = (__int64 *)*a2;
      *a1 = a2;
      a1[3] = &v6[v8];
      a1[1] = v12;
      a1[2] = v10;
      return a1;
    }
    goto LABEL_7;
  }
  v14 = *((unsigned int *)a2 + 6);
  v6 = (__int64 *)a2[2];
  if ( (_DWORD)v14 )
  {
    v7 = v14 - 1;
    v9 = (v14 - 1) & (((unsigned int)a3 >> 4) ^ ((unsigned int)a3 >> 9));
    v10 = &v6[2 * v9];
    v11 = *v10;
    if ( *v10 == a3 )
    {
LABEL_6:
      v8 = 2 * v14;
      goto LABEL_3;
    }
LABEL_7:
    for ( i = 1; ; ++i )
    {
      if ( v11 == -4096 )
      {
        if ( v5 )
        {
          v16 = 16;
          goto LABEL_11;
        }
        v14 = *((unsigned int *)a2 + 6);
        goto LABEL_13;
      }
      v9 = v7 & (i + v9);
      v10 = &v6[2 * v9];
      v11 = *v10;
      if ( *v10 == a3 )
        break;
    }
    if ( !v5 )
    {
      v14 = *((unsigned int *)a2 + 6);
      goto LABEL_6;
    }
    v8 = 16;
    goto LABEL_3;
  }
LABEL_13:
  v16 = 2 * v14;
LABEL_11:
  v17 = &v6[v16];
  v18 = (__int64 *)*a2;
  *a1 = a2;
  a1[2] = v17;
  a1[1] = v18;
  a1[3] = v17;
  return a1;
}
