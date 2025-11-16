// Function: sub_2B5A300
// Address: 0x2b5a300
//
__int64 __fastcall sub_2B5A300(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rbx
  int v9; // eax
  int v10; // esi
  __int64 v11; // rcx
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r12
  int v16; // eax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  int v20; // eax
  __int64 v21; // rax
  unsigned int v22; // esi
  __int64 v23; // r9
  int v24; // r11d
  __int64 *v25; // rdx
  unsigned int v26; // edi
  __int64 *v27; // rax
  __int64 v28; // r8
  __int64 *v29; // rax
  __int64 result; // rax
  int v31; // edx
  int v32; // eax
  int v33; // edi
  int v34; // r8d
  int v35; // r8d
  __int64 v36; // r9
  unsigned int v37; // eax
  __int64 v38; // r11
  int v39; // esi
  __int64 *v40; // rcx
  int v41; // r9d
  int v42; // r9d
  __int64 v43; // r10
  int v44; // esi
  unsigned int v45; // ecx
  __int64 *v46; // rax
  __int64 v47; // r8
  int v48; // r8d
  unsigned int v49; // [rsp+Ch] [rbp-44h]
  __int64 v50; // [rsp+10h] [rbp-40h]

  if ( a2 != a3 )
  {
    v7 = a2;
    v50 = a1 + 80;
    while ( 1 )
    {
      if ( sub_2B16010(v7) && (unsigned __int8)sub_2B099C0(v7) )
        goto LABEL_17;
      v9 = *(_DWORD *)(a1 + 104);
      if ( v9 )
      {
        v10 = v9 - 1;
        v11 = *(_QWORD *)(a1 + 88);
        v12 = (v9 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v13 = (__int64 *)(v11 + 16LL * v12);
        v14 = *v13;
        if ( *v13 == v7 )
        {
LABEL_5:
          v15 = v13[1];
          if ( v15 )
            goto LABEL_6;
        }
        else
        {
          v20 = 1;
          while ( v14 != -4096 )
          {
            v48 = v20 + 1;
            v12 = v10 & (v20 + v12);
            v13 = (__int64 *)(v11 + 16LL * v12);
            v14 = *v13;
            if ( *v13 == v7 )
              goto LABEL_5;
            v20 = v48;
          }
        }
      }
      v21 = sub_2B48820(a1);
      v22 = *(_DWORD *)(a1 + 104);
      v15 = v21;
      if ( !v22 )
        break;
      v23 = *(_QWORD *)(a1 + 88);
      v24 = 1;
      v25 = 0;
      v26 = (v22 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v27 = (__int64 *)(v23 + 16LL * v26);
      v28 = *v27;
      if ( *v27 != v7 )
      {
        while ( v28 != -4096 )
        {
          if ( v28 == -8192 && !v25 )
            v25 = v27;
          v26 = (v22 - 1) & (v24 + v26);
          v27 = (__int64 *)(v23 + 16LL * v26);
          v28 = *v27;
          if ( *v27 == v7 )
            goto LABEL_27;
          ++v24;
        }
        if ( !v25 )
          v25 = v27;
        v32 = *(_DWORD *)(a1 + 96);
        ++*(_QWORD *)(a1 + 80);
        v33 = v32 + 1;
        if ( 4 * (v32 + 1) < 3 * v22 )
        {
          if ( v22 - *(_DWORD *)(a1 + 100) - v33 <= v22 >> 3 )
          {
            v49 = ((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4);
            sub_2B5A120(v50, v22);
            v41 = *(_DWORD *)(a1 + 104);
            if ( !v41 )
            {
LABEL_86:
              ++*(_DWORD *)(a1 + 96);
              BUG();
            }
            v42 = v41 - 1;
            v43 = *(_QWORD *)(a1 + 88);
            v44 = 1;
            v45 = v42 & v49;
            v33 = *(_DWORD *)(a1 + 96) + 1;
            v46 = 0;
            v25 = (__int64 *)(v43 + 16LL * (v42 & v49));
            v47 = *v25;
            if ( *v25 != v7 )
            {
              while ( v47 != -4096 )
              {
                if ( v47 == -8192 && !v46 )
                  v46 = v25;
                v45 = v42 & (v44 + v45);
                v25 = (__int64 *)(v43 + 16LL * v45);
                v47 = *v25;
                if ( *v25 == v7 )
                  goto LABEL_57;
                ++v44;
              }
              if ( v46 )
                v25 = v46;
            }
          }
          goto LABEL_57;
        }
LABEL_61:
        sub_2B5A120(v50, 2 * v22);
        v34 = *(_DWORD *)(a1 + 104);
        if ( !v34 )
          goto LABEL_86;
        v35 = v34 - 1;
        v36 = *(_QWORD *)(a1 + 88);
        v37 = v35 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v33 = *(_DWORD *)(a1 + 96) + 1;
        v25 = (__int64 *)(v36 + 16LL * v37);
        v38 = *v25;
        if ( *v25 != v7 )
        {
          v39 = 1;
          v40 = 0;
          while ( v38 != -4096 )
          {
            if ( v38 == -8192 && !v40 )
              v40 = v25;
            v37 = v35 & (v39 + v37);
            v25 = (__int64 *)(v36 + 16LL * v37);
            v38 = *v25;
            if ( *v25 == v7 )
              goto LABEL_57;
            ++v39;
          }
          if ( v40 )
            v25 = v40;
        }
LABEL_57:
        *(_DWORD *)(a1 + 96) = v33;
        if ( *v25 != -4096 )
          --*(_DWORD *)(a1 + 100);
        *v25 = v7;
        v29 = v25 + 1;
        v25[1] = 0;
        goto LABEL_28;
      }
LABEL_27:
      v29 = v27 + 1;
LABEL_28:
      *v29 = v15;
LABEL_6:
      v16 = *(_DWORD *)(a1 + 204);
      *(_QWORD *)(v15 + 16) = v15;
      *(_QWORD *)(v15 + 24) = 0;
      *(_QWORD *)(v15 + 32) = 0;
      *(_DWORD *)(v15 + 136) = v16;
      *(_QWORD *)(v15 + 144) = -1;
      *(_BYTE *)(v15 + 152) = 0;
      *(_DWORD *)(v15 + 48) = 0;
      *(_DWORD *)(v15 + 96) = 0;
      *(_QWORD *)v15 = v7;
      *(_QWORD *)(v15 + 8) = 0;
      if ( (unsigned __int8)sub_B46420(v7) || (unsigned __int8)sub_B46490(v7) )
      {
        if ( *(_BYTE *)v7 == 85 )
        {
          v17 = *(_QWORD *)(v7 - 32);
          if ( v17 )
          {
            if ( !*(_BYTE *)v17 && *(_QWORD *)(v17 + 24) == *(_QWORD *)(v7 + 80) && (*(_BYTE *)(v17 + 33) & 0x20) != 0 )
            {
              v31 = *(_DWORD *)(v17 + 36);
              if ( v31 == 324 || v31 == 291 )
              {
LABEL_14:
                if ( !*(_BYTE *)v17 )
                {
                  if ( (v18 = *(_QWORD *)(v17 + 24), v18 == *(_QWORD *)(v7 + 80)) && *(_DWORD *)(v17 + 36) == 343
                    || v18 == *(_QWORD *)(v7 + 80) && *(_DWORD *)(v17 + 36) == 342 )
                  {
                    *(_BYTE *)(a1 + 192) = 1;
                  }
                }
                goto LABEL_17;
              }
            }
          }
        }
        if ( a4 )
          *(_QWORD *)(a4 + 32) = v15;
        else
          *(_QWORD *)(a1 + 176) = v15;
        a4 = v15;
      }
      if ( *(_BYTE *)v7 == 85 )
      {
        v17 = *(_QWORD *)(v7 - 32);
        if ( v17 )
          goto LABEL_14;
      }
LABEL_17:
      v19 = *(_QWORD *)(v7 + 32);
      if ( v19 == *(_QWORD *)(v7 + 40) + 48LL || !v19 )
      {
        v7 = 0;
        if ( !a3 )
          goto LABEL_30;
      }
      else
      {
        v7 = v19 - 24;
        if ( a3 == v19 - 24 )
          goto LABEL_30;
      }
    }
    ++*(_QWORD *)(a1 + 80);
    goto LABEL_61;
  }
LABEL_30:
  result = a5;
  if ( a5 )
  {
    if ( a4 )
      *(_QWORD *)(a4 + 32) = a5;
  }
  else
  {
    *(_QWORD *)(a1 + 184) = a4;
  }
  return result;
}
