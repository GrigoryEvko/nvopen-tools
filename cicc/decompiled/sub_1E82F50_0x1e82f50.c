// Function: sub_1E82F50
// Address: 0x1e82f50
//
__int64 __fastcall sub_1E82F50(__int64 a1, __int64 a2)
{
  int *v2; // rbx
  int *v3; // r14
  __int64 v6; // r12
  __int64 v7; // rax
  unsigned int v8; // edx
  unsigned int v9; // ecx
  unsigned int v10; // esi
  unsigned int v11; // edx
  __int64 v12; // r10
  unsigned int v13; // r9d
  __int64 *v14; // rax
  __int64 v15; // r8
  __int64 *v17; // rdi
  int v18; // eax
  int v19; // eax
  int v20; // esi
  int v21; // esi
  __int64 v22; // r9
  unsigned int v23; // ecx
  __int64 v24; // r8
  int v25; // r10d
  __int64 *v26; // r11
  int v27; // esi
  int v28; // esi
  __int64 v29; // r9
  int v30; // r10d
  unsigned int v31; // ecx
  __int64 v32; // r8
  unsigned int v33; // [rsp+8h] [rbp-48h]
  __int64 *v34; // [rsp+8h] [rbp-48h]
  __int64 v35; // [rsp+10h] [rbp-40h]
  int v36; // [rsp+18h] [rbp-38h]
  unsigned int v37; // [rsp+18h] [rbp-38h]
  unsigned int v38; // [rsp+18h] [rbp-38h]
  unsigned int v39; // [rsp+1Ch] [rbp-34h]

  v2 = *(int **)(a2 + 40);
  v3 = &v2[2 * *(unsigned int *)(a2 + 48)];
  if ( v2 != v3 )
  {
    v39 = 0;
    v35 = a1 + 376;
    while ( 1 )
    {
      while ( 1 )
      {
        if ( *v2 < 0 )
        {
          v6 = sub_1E69D00(*(_QWORD *)(*(_QWORD *)(a1 + 440) + 256LL), *v2);
          v7 = *(_QWORD *)(a1 + 8) + 88LL * *(int *)(*(_QWORD *)(v6 + 24) + 48LL);
          v8 = *(_DWORD *)(v7 + 24);
          if ( v8 != -1 )
          {
            v9 = *(_DWORD *)(a2 + 24);
            if ( v9 != -1 && *(_DWORD *)(v7 + 16) == *(_DWORD *)(a2 + 16) && *(_BYTE *)(v7 + 32) && v8 <= v9 )
              break;
          }
        }
        v2 += 2;
        if ( v3 == v2 )
          return v39;
      }
      v10 = *(_DWORD *)(a1 + 400);
      v11 = v2[1];
      if ( v10 )
      {
        v12 = *(_QWORD *)(a1 + 384);
        v13 = (v10 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v14 = (__int64 *)(v12 + 16LL * v13);
        v15 = *v14;
        if ( v6 == *v14 )
        {
          v11 += *((_DWORD *)v14 + 2);
          goto LABEL_13;
        }
        v36 = 1;
        v17 = 0;
        while ( v15 != -8 )
        {
          if ( v15 != -16 || v17 )
            v14 = v17;
          v13 = (v10 - 1) & (v36 + v13);
          v34 = (__int64 *)(v12 + 16LL * v13);
          v15 = *v34;
          if ( v6 == *v34 )
          {
            v11 += *((_DWORD *)v34 + 2);
            goto LABEL_13;
          }
          ++v36;
          v17 = v14;
          v14 = (__int64 *)(v12 + 16LL * v13);
        }
        if ( !v17 )
          v17 = v14;
        v18 = *(_DWORD *)(a1 + 392);
        ++*(_QWORD *)(a1 + 376);
        v19 = v18 + 1;
        if ( 4 * v19 < 3 * v10 )
        {
          if ( v10 - *(_DWORD *)(a1 + 396) - v19 > v10 >> 3 )
            goto LABEL_24;
          v33 = v11;
          v38 = ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4);
          sub_1E821A0(v35, v10);
          v27 = *(_DWORD *)(a1 + 400);
          if ( !v27 )
          {
LABEL_54:
            ++*(_DWORD *)(a1 + 392);
            BUG();
          }
          v28 = v27 - 1;
          v29 = *(_QWORD *)(a1 + 384);
          v26 = 0;
          v11 = v33;
          v30 = 1;
          v31 = v28 & v38;
          v19 = *(_DWORD *)(a1 + 392) + 1;
          v17 = (__int64 *)(v29 + 16LL * (v28 & v38));
          v32 = *v17;
          if ( v6 == *v17 )
            goto LABEL_24;
          while ( v32 != -8 )
          {
            if ( !v26 && v32 == -16 )
              v26 = v17;
            v31 = v28 & (v30 + v31);
            v17 = (__int64 *)(v29 + 16LL * v31);
            v32 = *v17;
            if ( v6 == *v17 )
              goto LABEL_24;
            ++v30;
          }
          goto LABEL_32;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 376);
      }
      v37 = v11;
      sub_1E821A0(v35, 2 * v10);
      v20 = *(_DWORD *)(a1 + 400);
      if ( !v20 )
        goto LABEL_54;
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 384);
      v11 = v37;
      v23 = v21 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v19 = *(_DWORD *)(a1 + 392) + 1;
      v17 = (__int64 *)(v22 + 16LL * v23);
      v24 = *v17;
      if ( v6 == *v17 )
        goto LABEL_24;
      v25 = 1;
      v26 = 0;
      while ( v24 != -8 )
      {
        if ( !v26 && v24 == -16 )
          v26 = v17;
        v23 = v21 & (v25 + v23);
        v17 = (__int64 *)(v22 + 16LL * v23);
        v24 = *v17;
        if ( v6 == *v17 )
          goto LABEL_24;
        ++v25;
      }
LABEL_32:
      if ( v26 )
        v17 = v26;
LABEL_24:
      *(_DWORD *)(a1 + 392) = v19;
      if ( *v17 != -8 )
        --*(_DWORD *)(a1 + 396);
      *v17 = v6;
      v17[1] = 0;
LABEL_13:
      if ( v39 >= v11 )
        v11 = v39;
      v2 += 2;
      v39 = v11;
      if ( v3 == v2 )
        return v39;
    }
  }
  return 0;
}
