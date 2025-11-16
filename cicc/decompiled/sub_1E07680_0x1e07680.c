// Function: sub_1E07680
// Address: 0x1e07680
//
__int64 *__fastcall sub_1E07680(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 *result; // rax
  __int64 v6; // rbx
  __int64 v7; // r11
  unsigned int v8; // esi
  __int64 v9; // r9
  __int64 v10; // rdx
  unsigned int v11; // r10d
  char *v12; // rdi
  int v13; // esi
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rdi
  __int64 v18; // r15
  __int64 v19; // rdi
  int v20; // ecx
  int v21; // eax
  __int64 v22; // [rsp+8h] [rbp-78h]
  __int64 v23; // [rsp+10h] [rbp-70h]
  __int64 v24; // [rsp+18h] [rbp-68h]
  __int64 v25; // [rsp+18h] [rbp-68h]
  __int64 v26; // [rsp+28h] [rbp-58h]
  __int64 v27; // [rsp+38h] [rbp-48h] BYREF
  __int64 v28; // [rsp+40h] [rbp-40h] BYREF
  char *v29[7]; // [rsp+48h] [rbp-38h] BYREF

  v22 = a1 + 24;
  sub_1E071E0(a1 + 24, (__int64 *)(*(_QWORD *)a1 + 8LL))[4] = *a3;
  result = *(__int64 **)a1;
  v26 = (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 3;
  if ( v26 != 1 )
  {
    v23 = a2 + 24;
    v6 = 1;
    while ( 1 )
    {
      v10 = result[v6];
      v11 = *(_DWORD *)(a2 + 48);
      v27 = v10;
      if ( !v11 )
        break;
      v7 = *(_QWORD *)(a2 + 32);
      v8 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      result = (__int64 *)(v7 + 16LL * v8);
      v9 = *result;
      if ( v10 != *result )
      {
        v20 = 1;
        v12 = 0;
        while ( v9 != -8 )
        {
          if ( v9 == -16 && !v12 )
            v12 = (char *)result;
          v8 = (v11 - 1) & (v20 + v8);
          result = (__int64 *)(v7 + 16LL * v8);
          v9 = *result;
          if ( v10 == *result )
            goto LABEL_4;
          ++v20;
        }
        if ( !v12 )
          v12 = (char *)result;
        v21 = *(_DWORD *)(a2 + 40);
        ++*(_QWORD *)(a2 + 24);
        v13 = v21 + 1;
        if ( 4 * (v21 + 1) < 3 * v11 )
        {
          if ( v11 - *(_DWORD *)(a2 + 44) - v13 <= v11 >> 3 )
          {
            sub_1E06190(v23, v11);
LABEL_10:
            sub_1E05F80(v23, &v27, v29);
            v12 = v29[0];
            v10 = v27;
            v13 = *(_DWORD *)(a2 + 40) + 1;
          }
          *(_DWORD *)(a2 + 40) = v13;
          if ( *(_QWORD *)v12 != -8 )
            --*(_DWORD *)(a2 + 44);
          *(_QWORD *)v12 = v10;
          *((_QWORD *)v12 + 1) = 0;
          goto LABEL_14;
        }
LABEL_9:
        sub_1E06190(v23, 2 * v11);
        goto LABEL_10;
      }
LABEL_4:
      if ( result[1] )
        goto LABEL_5;
LABEL_14:
      v28 = v27;
      v14 = 0;
      if ( (unsigned __int8)sub_1E060E0(v22, &v28, v29)
        && v29[0] != (char *)(*(_QWORD *)(a1 + 32) + 72LL * *(unsigned int *)(a1 + 48)) )
      {
        v14 = *((_QWORD *)v29[0] + 4);
      }
      v24 = sub_1E064E0(a1, v14, a2);
      sub_1E04AB0(&v28, v27, v24);
      v29[0] = (char *)v28;
      sub_1E06030(v24 + 24, v29);
      v15 = v28;
      v28 = 0;
      v25 = v15;
      result = sub_1E063B0(v23, &v27);
      v16 = result[1];
      result[1] = v25;
      if ( v16 )
      {
        v17 = *(_QWORD *)(v16 + 24);
        if ( v17 )
          j_j___libc_free_0(v17, *(_QWORD *)(v16 + 40) - v17);
        result = (__int64 *)j_j___libc_free_0(v16, 56);
      }
      v18 = v28;
      if ( !v28 )
      {
LABEL_5:
        if ( v26 == ++v6 )
          return result;
        goto LABEL_6;
      }
      v19 = *(_QWORD *)(v28 + 24);
      if ( v19 )
        j_j___libc_free_0(v19, *(_QWORD *)(v28 + 40) - v19);
      ++v6;
      result = (__int64 *)j_j___libc_free_0(v18, 56);
      if ( v26 == v6 )
        return result;
LABEL_6:
      result = *(__int64 **)a1;
    }
    ++*(_QWORD *)(a2 + 24);
    goto LABEL_9;
  }
  return result;
}
