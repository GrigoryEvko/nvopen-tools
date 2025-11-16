// Function: sub_3917930
// Address: 0x3917930
//
unsigned __int64 __fastcall sub_3917930(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  unsigned __int64 result; // rax
  __int64 *v6; // r13
  __int64 v8; // r9
  unsigned int v9; // edi
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r12
  __int64 v13; // r15
  unsigned int v14; // esi
  unsigned __int64 v15; // rcx
  int v16; // edi
  int v17; // edi
  __int64 v18; // r10
  unsigned int v19; // esi
  int v20; // edx
  __int64 v21; // r9
  int v22; // r11d
  __int64 *v23; // r8
  int v24; // edx
  int v25; // esi
  int v26; // esi
  __int64 v27; // r9
  __int64 *v28; // r10
  unsigned int v29; // r12d
  int v30; // r8d
  __int64 v31; // rdi
  int v32; // r8d
  __int64 *v33; // r11
  __int64 v34; // [rsp+8h] [rbp-48h]
  __int64 *v35; // [rsp+10h] [rbp-40h]
  unsigned __int64 v36; // [rsp+18h] [rbp-38h]
  unsigned __int64 v37; // [rsp+18h] [rbp-38h]
  unsigned __int64 v38; // [rsp+18h] [rbp-38h]

  v3 = *(_QWORD *)(a3 + 8);
  v34 = a1 + 80;
  result = v3 + 8LL * *(unsigned int *)(a3 + 16);
  v35 = (__int64 *)result;
  if ( result != v3 )
  {
    v6 = *(__int64 **)(a3 + 8);
    result = 0;
    while ( 1 )
    {
      v13 = *v6;
      v14 = *(_DWORD *)(a1 + 104);
      v15 = *(unsigned int *)(*v6 + 24) * ((*(unsigned int *)(*v6 + 24) + result - 1) / *(unsigned int *)(*v6 + 24));
      if ( !v14 )
        break;
      v8 = *(_QWORD *)(a1 + 88);
      v9 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( v13 != *v10 )
      {
        v22 = 1;
        v23 = 0;
        while ( v11 != -8 )
        {
          if ( !v23 && v11 == -16 )
            v23 = v10;
          v9 = (v14 - 1) & (v22 + v9);
          v10 = (__int64 *)(v8 + 16LL * v9);
          v11 = *v10;
          if ( v13 == *v10 )
            goto LABEL_4;
          ++v22;
        }
        v24 = *(_DWORD *)(a1 + 96);
        if ( v23 )
          v10 = v23;
        ++*(_QWORD *)(a1 + 80);
        v20 = v24 + 1;
        if ( 4 * v20 < 3 * v14 )
        {
          if ( v14 - *(_DWORD *)(a1 + 100) - v20 <= v14 >> 3 )
          {
            v38 = v15;
            sub_3917770(v34, v14);
            v25 = *(_DWORD *)(a1 + 104);
            if ( !v25 )
            {
LABEL_45:
              ++*(_DWORD *)(a1 + 96);
              BUG();
            }
            v26 = v25 - 1;
            v27 = *(_QWORD *)(a1 + 88);
            v28 = 0;
            v29 = v26 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v30 = 1;
            v20 = *(_DWORD *)(a1 + 96) + 1;
            v15 = v38;
            v10 = (__int64 *)(v27 + 16LL * v29);
            v31 = *v10;
            if ( v13 != *v10 )
            {
              while ( v31 != -8 )
              {
                if ( !v28 && v31 == -16 )
                  v28 = v10;
                v29 = v26 & (v30 + v29);
                v10 = (__int64 *)(v27 + 16LL * v29);
                v31 = *v10;
                if ( v13 == *v10 )
                  goto LABEL_9;
                ++v30;
              }
              if ( v28 )
                v10 = v28;
            }
          }
          goto LABEL_9;
        }
LABEL_7:
        v37 = v15;
        sub_3917770(v34, 2 * v14);
        v16 = *(_DWORD *)(a1 + 104);
        if ( !v16 )
          goto LABEL_45;
        v17 = v16 - 1;
        v18 = *(_QWORD *)(a1 + 88);
        v19 = v17 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v20 = *(_DWORD *)(a1 + 96) + 1;
        v15 = v37;
        v10 = (__int64 *)(v18 + 16LL * v19);
        v21 = *v10;
        if ( v13 != *v10 )
        {
          v32 = 1;
          v33 = 0;
          while ( v21 != -8 )
          {
            if ( !v33 && v21 == -16 )
              v33 = v10;
            v19 = v17 & (v32 + v19);
            v10 = (__int64 *)(v18 + 16LL * v19);
            v21 = *v10;
            if ( v13 == *v10 )
              goto LABEL_9;
            ++v32;
          }
          if ( v33 )
            v10 = v33;
        }
LABEL_9:
        *(_DWORD *)(a1 + 96) = v20;
        if ( *v10 != -8 )
          --*(_DWORD *)(a1 + 100);
        *v10 = v13;
        v10[1] = 0;
      }
LABEL_4:
      v10[1] = v15;
      ++v6;
      v36 = v15;
      v12 = sub_38D04A0((_QWORD *)a3, v13);
      result = v36 + v12 + sub_39142B0(a1, v13, a3);
      if ( v35 == v6 )
        return result;
    }
    ++*(_QWORD *)(a1 + 80);
    goto LABEL_7;
  }
  return result;
}
