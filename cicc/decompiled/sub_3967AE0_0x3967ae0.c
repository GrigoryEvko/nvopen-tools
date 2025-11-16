// Function: sub_3967AE0
// Address: 0x3967ae0
//
__int64 *__fastcall sub_3967AE0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, int a5, __int64 a6)
{
  __int64 *v6; // r11
  __int64 **v8; // r13
  __int64 *result; // rax
  __int64 *v10; // r8
  __int64 **v11; // r15
  unsigned int v12; // edx
  __int64 v13; // rdi
  __int64 *v14; // r14
  __int64 v15; // rax
  __int64 v16; // r12
  unsigned int v17; // esi
  int v18; // ecx
  __int64 v19; // rdx
  __int64 *v20; // r11
  int v21; // ecx
  __int64 v22; // rax
  __int64 v23; // r12
  char v24; // al
  char v25; // r8
  __int64 *v26; // rax
  int v27; // ecx
  unsigned int v28; // esi
  int v29; // edx
  __int64 v30; // rdx
  __int64 *v31; // [rsp+10h] [rbp-60h]
  __int64 *v32; // [rsp+10h] [rbp-60h]
  int v33; // [rsp+10h] [rbp-60h]
  __int64 *v34; // [rsp+18h] [rbp-58h]
  __int64 *v35; // [rsp+28h] [rbp-48h]
  _QWORD v37[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = (__int64 *)a3;
  if ( *(_BYTE *)(a3 + 193) )
    goto LABEL_2;
  v22 = *(unsigned int *)(a2 + 152);
  if ( (unsigned int)v22 >= *(_DWORD *)(a2 + 156) )
  {
    sub_16CD150(a2 + 144, (const void *)(a2 + 160), 0, 8, a5, a6);
    v22 = *(unsigned int *)(a2 + 152);
    v6 = (__int64 *)a3;
  }
  v35 = v6;
  *(_QWORD *)(*(_QWORD *)(a2 + 144) + 8 * v22) = v6;
  ++*(_DWORD *)(a2 + 152);
  v23 = *a1;
  v24 = sub_1AFF330(*a1, v6, v37);
  v6 = v35;
  v25 = v24;
  v26 = (__int64 *)v37[0];
  if ( v25 )
  {
    *(_QWORD *)(v37[0] + 8LL) = *(_QWORD *)a2;
    goto LABEL_2;
  }
  v27 = *(_DWORD *)(v23 + 16);
  v28 = *(_DWORD *)(v23 + 24);
  ++*(_QWORD *)v23;
  v29 = v27 + 1;
  if ( 4 * (v27 + 1) >= 3 * v28 )
  {
    v28 *= 2;
LABEL_39:
    sub_1447B20(v23, v28);
    sub_1AFF330(v23, v35, v37);
    v26 = (__int64 *)v37[0];
    v6 = v35;
    v29 = *(_DWORD *)(v23 + 16) + 1;
    goto LABEL_30;
  }
  if ( v28 - *(_DWORD *)(v23 + 20) - v29 <= v28 >> 3 )
    goto LABEL_39;
LABEL_30:
  *(_DWORD *)(v23 + 16) = v29;
  if ( *v26 != -8 )
    --*(_DWORD *)(v23 + 20);
  v30 = *v6;
  v26[1] = 0;
  *v26 = v30;
  v26[1] = *(_QWORD *)a2;
LABEL_2:
  v8 = (__int64 **)v6[18];
  result = (__int64 *)(a2 + 160);
  if ( &v8[*((unsigned int *)v6 + 38)] != v8 )
  {
    v10 = a1;
    v34 = v6;
    v11 = &v8[*((unsigned int *)v6 + 38)];
    while ( 1 )
    {
      v14 = *v8;
      v15 = *(unsigned int *)(a2 + 152);
      if ( (unsigned int)v15 >= *(_DWORD *)(a2 + 156) )
      {
        v32 = v10;
        sub_16CD150(a2 + 144, (const void *)(a2 + 160), 0, 8, (int)v10, a6);
        v15 = *(unsigned int *)(a2 + 152);
        v10 = v32;
      }
      *(_QWORD *)(*(_QWORD *)(a2 + 144) + 8 * v15) = v14;
      ++*(_DWORD *)(a2 + 152);
      v16 = *v10;
      v17 = *(_DWORD *)(*v10 + 24);
      if ( !v17 )
        break;
      a6 = *(_QWORD *)(v16 + 8);
      v12 = (v17 - 1) & (((unsigned int)*v14 >> 9) ^ ((unsigned int)*v14 >> 4));
      result = (__int64 *)(a6 + 16LL * v12);
      v13 = *result;
      if ( *result == *v14 )
      {
LABEL_5:
        ++v8;
        result[1] = *(_QWORD *)a2;
        if ( v11 == v8 )
          goto LABEL_15;
      }
      else
      {
        v33 = 1;
        v20 = 0;
        while ( v13 != -8 )
        {
          if ( v13 == -16 && !v20 )
            v20 = result;
          v12 = (v17 - 1) & (v33 + v12);
          result = (__int64 *)(a6 + 16LL * v12);
          v13 = *result;
          if ( *v14 == *result )
            goto LABEL_5;
          ++v33;
        }
        v21 = *(_DWORD *)(v16 + 16);
        if ( v20 )
          result = v20;
        ++*(_QWORD *)v16;
        v18 = v21 + 1;
        if ( 4 * v18 >= 3 * v17 )
          goto LABEL_10;
        if ( v17 - *(_DWORD *)(v16 + 20) - v18 > v17 >> 3 )
          goto LABEL_12;
        v31 = v10;
LABEL_11:
        sub_1447B20(v16, v17);
        sub_1AFF330(v16, v14, v37);
        result = (__int64 *)v37[0];
        v10 = v31;
        v18 = *(_DWORD *)(v16 + 16) + 1;
LABEL_12:
        *(_DWORD *)(v16 + 16) = v18;
        if ( *result != -8 )
          --*(_DWORD *)(v16 + 20);
        v19 = *v14;
        result[1] = 0;
        ++v8;
        *result = v19;
        result[1] = *(_QWORD *)a2;
        if ( v11 == v8 )
        {
LABEL_15:
          v6 = v34;
          goto LABEL_16;
        }
      }
    }
    ++*(_QWORD *)v16;
LABEL_10:
    v31 = v10;
    v17 *= 2;
    goto LABEL_11;
  }
LABEL_16:
  *((_DWORD *)v6 + 38) = 0;
  *((_BYTE *)v6 + 192) = 1;
  return result;
}
