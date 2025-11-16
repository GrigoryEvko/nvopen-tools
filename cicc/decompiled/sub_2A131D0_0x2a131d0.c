// Function: sub_2A131D0
// Address: 0x2a131d0
//
__int64 __fastcall sub_2A131D0(__int64 *a1)
{
  __int64 result; // rax
  __int64 *v2; // r12
  __int64 v4; // r8
  int v5; // r15d
  __int64 *v6; // r11
  unsigned int v7; // edi
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 *v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rcx
  __int64 v13; // r9
  unsigned int v14; // esi
  __int64 v15; // rcx
  unsigned int v16; // esi
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // rdi
  int v20; // eax
  int v21; // edx
  __int64 v22; // rax
  int v23; // eax
  __int64 v24; // rdi
  int v25; // esi
  __int64 v26; // r8
  unsigned int v27; // ecx
  __int64 v28; // rax
  int v29; // r10d
  __int64 *v30; // r9
  int v31; // eax
  __int64 v32; // rdi
  int v33; // esi
  __int64 v34; // r8
  int v35; // r10d
  unsigned int v36; // ecx
  __int64 v37; // rax
  int v38; // eax
  int v39; // r8d

  result = *a1;
  *(_QWORD *)(*a1 + 32) = a1[1];
  v2 = (__int64 *)a1[2];
  if ( v2 )
  {
    while ( 1 )
    {
      v11 = *a1;
      v12 = v2[1];
      v13 = *(_QWORD *)(*a1 + 8);
      v14 = *(_DWORD *)(*a1 + 24);
      if ( !v12 )
      {
        if ( v14 )
        {
          v15 = v2[2];
          v16 = v14 - 1;
          v17 = v16 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
          v18 = (__int64 *)(v13 + 16LL * v17);
          v19 = *v18;
          if ( v15 == *v18 )
          {
LABEL_11:
            *v18 = -8192;
            --*(_DWORD *)(v11 + 16);
            ++*(_DWORD *)(v11 + 20);
          }
          else
          {
            v38 = 1;
            while ( v19 != -4096 )
            {
              v39 = v38 + 1;
              v17 = v16 & (v38 + v17);
              v18 = (__int64 *)(v13 + 16LL * v17);
              v19 = *v18;
              if ( v15 == *v18 )
                goto LABEL_11;
              v38 = v39;
            }
          }
        }
        goto LABEL_7;
      }
      if ( !v14 )
        break;
      v4 = v2[2];
      v5 = 1;
      v6 = 0;
      v7 = (v14 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v8 = (__int64 *)(v13 + 16LL * v7);
      v9 = *v8;
      if ( v4 != *v8 )
      {
        while ( v9 != -4096 )
        {
          if ( !v6 && v9 == -8192 )
            v6 = v8;
          v7 = (v14 - 1) & (v5 + v7);
          v8 = (__int64 *)(v13 + 16LL * v7);
          v9 = *v8;
          if ( v4 == *v8 )
            goto LABEL_5;
          ++v5;
        }
        if ( !v6 )
          v6 = v8;
        v20 = *(_DWORD *)(v11 + 16);
        ++*(_QWORD *)v11;
        v21 = v20 + 1;
        if ( 4 * (v20 + 1) < 3 * v14 )
        {
          if ( v14 - *(_DWORD *)(v11 + 20) - v21 <= v14 >> 3 )
          {
            sub_2A12FF0(v11, v14);
            v31 = *(_DWORD *)(v11 + 24);
            if ( !v31 )
            {
LABEL_51:
              ++*(_DWORD *)(v11 + 16);
              BUG();
            }
            v32 = v2[2];
            v33 = v31 - 1;
            v34 = *(_QWORD *)(v11 + 8);
            v30 = 0;
            v35 = 1;
            v21 = *(_DWORD *)(v11 + 16) + 1;
            v36 = (v31 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
            v6 = (__int64 *)(v34 + 16LL * v36);
            v37 = *v6;
            if ( *v6 != v32 )
            {
              while ( v37 != -4096 )
              {
                if ( v37 == -8192 && !v30 )
                  v30 = v6;
                v36 = v33 & (v35 + v36);
                v6 = (__int64 *)(v34 + 16LL * v36);
                v37 = *v6;
                if ( v32 == *v6 )
                  goto LABEL_23;
                ++v35;
              }
              goto LABEL_31;
            }
          }
          goto LABEL_23;
        }
LABEL_27:
        sub_2A12FF0(v11, 2 * v14);
        v23 = *(_DWORD *)(v11 + 24);
        if ( !v23 )
          goto LABEL_51;
        v24 = v2[2];
        v25 = v23 - 1;
        v26 = *(_QWORD *)(v11 + 8);
        v21 = *(_DWORD *)(v11 + 16) + 1;
        v27 = (v23 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v6 = (__int64 *)(v26 + 16LL * v27);
        v28 = *v6;
        if ( *v6 != v24 )
        {
          v29 = 1;
          v30 = 0;
          while ( v28 != -4096 )
          {
            if ( !v30 && v28 == -8192 )
              v30 = v6;
            v27 = v25 & (v29 + v27);
            v6 = (__int64 *)(v26 + 16LL * v27);
            v28 = *v6;
            if ( v24 == *v6 )
              goto LABEL_23;
            ++v29;
          }
LABEL_31:
          if ( v30 )
            v6 = v30;
        }
LABEL_23:
        *(_DWORD *)(v11 + 16) = v21;
        if ( *v6 != -4096 )
          --*(_DWORD *)(v11 + 20);
        v22 = v2[2];
        v6[1] = 0;
        *v6 = v22;
        v10 = v6 + 1;
        v12 = v2[1];
        goto LABEL_6;
      }
LABEL_5:
      v10 = v8 + 1;
LABEL_6:
      *v10 = v12;
LABEL_7:
      a1[2] = *v2;
      result = sub_C7D6A0((__int64)v2, 40, 8);
      v2 = (__int64 *)a1[2];
      if ( !v2 )
        return result;
    }
    ++*(_QWORD *)v11;
    goto LABEL_27;
  }
  return result;
}
