// Function: sub_1E604F0
// Address: 0x1e604f0
//
__int64 *__fastcall sub_1E604F0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 *result; // rax
  __int64 v6; // r13
  __int64 v7; // r11
  unsigned int v8; // esi
  __int64 v9; // r10
  __int64 v10; // rdx
  unsigned int v11; // r9d
  __int64 *v12; // rdi
  int v13; // esi
  __int64 v14; // rsi
  __int64 v15; // rdx
  _BYTE *v16; // rsi
  __int64 v17; // r12
  __int64 v18; // rdi
  __int64 v19; // r12
  __int64 v20; // rdi
  int v21; // ecx
  int v22; // eax
  __int64 v23; // [rsp+8h] [rbp-78h]
  __int64 v24; // [rsp+10h] [rbp-70h]
  __int64 v25; // [rsp+18h] [rbp-68h]
  __int64 v26; // [rsp+18h] [rbp-68h]
  __int64 v27; // [rsp+28h] [rbp-58h]
  __int64 v28; // [rsp+38h] [rbp-48h] BYREF
  __int64 v29; // [rsp+40h] [rbp-40h] BYREF
  _QWORD v30[7]; // [rsp+48h] [rbp-38h] BYREF

  v23 = a1 + 24;
  sub_1E60050(a1 + 24, (__int64 *)(*(_QWORD *)a1 + 8LL))[4] = *a3;
  result = *(__int64 **)a1;
  v27 = (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 3;
  if ( v27 != 1 )
  {
    v24 = a2 + 48;
    v6 = 1;
    while ( 1 )
    {
      v10 = result[v6];
      v11 = *(_DWORD *)(a2 + 72);
      v28 = v10;
      if ( !v11 )
        break;
      v7 = *(_QWORD *)(a2 + 56);
      v8 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      result = (__int64 *)(v7 + 16LL * v8);
      v9 = *result;
      if ( v10 != *result )
      {
        v21 = 1;
        v12 = 0;
        while ( v9 != -8 )
        {
          if ( !v12 && v9 == -16 )
            v12 = result;
          v8 = (v11 - 1) & (v21 + v8);
          result = (__int64 *)(v7 + 16LL * v8);
          v9 = *result;
          if ( v10 == *result )
            goto LABEL_4;
          ++v21;
        }
        if ( !v12 )
          v12 = result;
        v22 = *(_DWORD *)(a2 + 64);
        ++*(_QWORD *)(a2 + 48);
        v13 = v22 + 1;
        if ( 4 * (v22 + 1) < 3 * v11 )
        {
          if ( v11 - *(_DWORD *)(a2 + 68) - v13 <= v11 >> 3 )
          {
            sub_1E06190(v24, v11);
LABEL_10:
            sub_1E05F80(v24, &v28, v30);
            v12 = (__int64 *)v30[0];
            v10 = v28;
            v13 = *(_DWORD *)(a2 + 64) + 1;
          }
          *(_DWORD *)(a2 + 64) = v13;
          if ( *v12 != -8 )
            --*(_DWORD *)(a2 + 68);
          *v12 = v10;
          v12[1] = 0;
          goto LABEL_14;
        }
LABEL_9:
        sub_1E06190(v24, 2 * v11);
        goto LABEL_10;
      }
LABEL_4:
      if ( result[1] )
        goto LABEL_5;
LABEL_14:
      v29 = v28;
      v14 = 0;
      if ( (unsigned __int8)sub_1E5F140(v23, &v29, v30)
        && v30[0] != *(_QWORD *)(a1 + 32) + 72LL * *(unsigned int *)(a1 + 48) )
      {
        v14 = *(_QWORD *)(v30[0] + 32LL);
      }
      v25 = sub_1E5F4B0(a1, v14, a2);
      sub_1E5E730(&v29, v28, v25);
      v15 = v29;
      v30[0] = v29;
      v16 = *(_BYTE **)(v25 + 32);
      if ( v16 == *(_BYTE **)(v25 + 40) )
      {
        sub_1D82C90(v25 + 24, v16, v30);
        v15 = v29;
      }
      else
      {
        if ( v16 )
        {
          *(_QWORD *)v16 = v29;
          v16 = *(_BYTE **)(v25 + 32);
          v15 = v29;
        }
        *(_QWORD *)(v25 + 32) = v16 + 8;
      }
      v26 = v15;
      v29 = 0;
      result = sub_1E063B0(v24, &v28);
      v17 = result[1];
      result[1] = v26;
      if ( v17 )
      {
        v18 = *(_QWORD *)(v17 + 24);
        if ( v18 )
          j_j___libc_free_0(v18, *(_QWORD *)(v17 + 40) - v18);
        result = (__int64 *)j_j___libc_free_0(v17, 56);
      }
      v19 = v29;
      if ( !v29 )
      {
LABEL_5:
        if ( v27 == ++v6 )
          return result;
        goto LABEL_6;
      }
      v20 = *(_QWORD *)(v29 + 24);
      if ( v20 )
        j_j___libc_free_0(v20, *(_QWORD *)(v29 + 40) - v20);
      ++v6;
      result = (__int64 *)j_j___libc_free_0(v19, 56);
      if ( v27 == v6 )
        return result;
LABEL_6:
      result = *(__int64 **)a1;
    }
    ++*(_QWORD *)(a2 + 48);
    goto LABEL_9;
  }
  return result;
}
