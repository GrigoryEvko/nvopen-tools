// Function: sub_28CA760
// Address: 0x28ca760
//
void __fastcall sub_28CA760(__int64 a1, __int64 a2)
{
  __int64 i; // rsi
  __int64 v5; // r10
  int v6; // edx
  unsigned int v7; // ecx
  __int64 v8; // rax
  unsigned __int8 *v9; // r11
  unsigned int v10; // ecx
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int8 *v13; // rcx
  __int64 v14; // r9
  int v15; // edx
  int v16; // edx
  unsigned int v17; // r10d
  unsigned __int8 *v18; // r11
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // rsi
  unsigned int v22; // edx
  __int64 *v23; // r12
  __int64 v24; // rcx
  __int64 v25; // rax
  unsigned __int8 **v26; // rdx
  bool v27; // zf
  __int64 v28; // rdx
  unsigned __int8 **v29; // rcx
  unsigned __int8 *v30; // rcx
  __int64 v31; // rdi
  int v32; // edx
  __int64 v33; // r8
  int v34; // edx
  unsigned int v35; // ecx
  __int64 v36; // rax
  unsigned __int8 *v37; // r9
  unsigned int v38; // ecx
  __int64 v39; // rdx
  __int64 v40; // rax
  unsigned __int8 **v41; // rax
  int v42; // edx
  unsigned int v43; // r8d
  unsigned __int8 *v44; // r9
  int v45; // eax
  int v46; // eax
  int v47; // r12d
  int v48; // eax
  int v49; // r10d
  int v50; // r8d
  int v51; // r12d
  int v52; // r10d
  unsigned __int8 **v53; // [rsp-68h] [rbp-68h] BYREF
  unsigned __int8 **v54; // [rsp-60h] [rbp-60h]
  __int64 *v55; // [rsp-58h] [rbp-58h]
  __int64 v56; // [rsp-50h] [rbp-50h]
  _QWORD v57[9]; // [rsp-48h] [rbp-48h] BYREF

  if ( *(_BYTE *)a2 == 26 )
    return;
  for ( i = *(_QWORD *)(a2 + 16); i; i = *(_QWORD *)(i + 8) )
  {
    v13 = *(unsigned __int8 **)(i + 24);
    v14 = *(_QWORD *)(a1 + 2424);
    v15 = *(_DWORD *)(a1 + 2440);
    if ( (unsigned int)*v13 - 26 <= 1 )
    {
      v5 = *((_QWORD *)v13 + 9);
      if ( !v15 )
        goto LABEL_13;
      v6 = v15 - 1;
      v7 = v6 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v8 = v14 + 16LL * v7;
      v9 = *(unsigned __int8 **)v8;
      if ( v5 != *(_QWORD *)v8 )
      {
        v46 = 1;
        while ( v9 != (unsigned __int8 *)-4096LL )
        {
          v47 = v46 + 1;
          v7 = v6 & (v46 + v7);
          v8 = v14 + 16LL * v7;
          v9 = *(unsigned __int8 **)v8;
          if ( v5 == *(_QWORD *)v8 )
            goto LABEL_6;
          v46 = v47;
        }
        goto LABEL_13;
      }
    }
    else
    {
      if ( !v15 )
        goto LABEL_13;
      v16 = v15 - 1;
      v17 = v16 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v8 = v14 + 16LL * v17;
      v18 = *(unsigned __int8 **)v8;
      if ( v13 != *(unsigned __int8 **)v8 )
      {
        v19 = 1;
        while ( v18 != (unsigned __int8 *)-4096LL )
        {
          v51 = v19 + 1;
          v17 = v16 & (v19 + v17);
          v8 = v14 + 16LL * v17;
          v18 = *(unsigned __int8 **)v8;
          if ( v13 == *(unsigned __int8 **)v8 )
            goto LABEL_6;
          v19 = v51;
        }
LABEL_13:
        v11 = 1;
        v12 = 0;
        goto LABEL_7;
      }
    }
LABEL_6:
    v10 = *(_DWORD *)(v8 + 8);
    v11 = 1LL << v10;
    v12 = 8LL * (v10 >> 6);
LABEL_7:
    *(_QWORD *)(*(_QWORD *)(a1 + 2280) + v12) |= v11;
  }
  v20 = *(unsigned int *)(a1 + 1912);
  v21 = *(_QWORD *)(a1 + 1896);
  if ( (_DWORD)v20 )
  {
    v22 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v23 = (__int64 *)(v21 + 56LL * v22);
    v24 = *v23;
    if ( *v23 == a2 )
    {
LABEL_16:
      if ( v23 != (__int64 *)(v21 + 56 * v20) )
      {
        v25 = v23[2];
        if ( *((_BYTE *)v23 + 36) )
          v26 = (unsigned __int8 **)(v25 + 8LL * *((unsigned int *)v23 + 7));
        else
          v26 = (unsigned __int8 **)(v25 + 8LL * *((unsigned int *)v23 + 6));
        v53 = (unsigned __int8 **)v23[2];
        v54 = v26;
        sub_254BBF0((__int64)&v53);
        v55 = v23 + 1;
        v27 = *((_BYTE *)v23 + 36) == 0;
        v56 = v23[1];
        if ( v27 )
          v28 = *((unsigned int *)v23 + 6);
        else
          v28 = *((unsigned int *)v23 + 7);
        v57[0] = v23[2] + 8 * v28;
        v57[1] = v57[0];
        sub_254BBF0((__int64)v57);
        v57[2] = v23 + 1;
        v29 = v53;
        v57[3] = v23[1];
        if ( (unsigned __int8 **)v57[0] != v53 )
        {
          while ( 1 )
          {
            v30 = *v29;
            v31 = *(_QWORD *)(a1 + 2424);
            v32 = *(_DWORD *)(a1 + 2440);
            if ( (unsigned int)*v30 - 26 > 1 )
            {
              if ( !v32 )
                goto LABEL_38;
              v42 = v32 - 1;
              v43 = v42 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
              v36 = v31 + 16LL * v43;
              v44 = *(unsigned __int8 **)v36;
              if ( *(unsigned __int8 **)v36 != v30 )
              {
                v45 = 1;
                while ( v44 != (unsigned __int8 *)-4096LL )
                {
                  v52 = v45 + 1;
                  v43 = v42 & (v45 + v43);
                  v36 = v31 + 16LL * v43;
                  v44 = *(unsigned __int8 **)v36;
                  if ( v30 == *(unsigned __int8 **)v36 )
                    goto LABEL_25;
                  v45 = v52;
                }
                goto LABEL_38;
              }
            }
            else
            {
              v33 = *((_QWORD *)v30 + 9);
              if ( !v32 )
                goto LABEL_38;
              v34 = v32 - 1;
              v35 = v34 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
              v36 = v31 + 16LL * v35;
              v37 = *(unsigned __int8 **)v36;
              if ( v33 != *(_QWORD *)v36 )
              {
                v48 = 1;
                while ( v37 != (unsigned __int8 *)-4096LL )
                {
                  v49 = v48 + 1;
                  v35 = v34 & (v48 + v35);
                  v36 = v31 + 16LL * v35;
                  v37 = *(unsigned __int8 **)v36;
                  if ( v33 == *(_QWORD *)v36 )
                    goto LABEL_25;
                  v48 = v49;
                }
LABEL_38:
                v39 = 1;
                v40 = 0;
                goto LABEL_26;
              }
            }
LABEL_25:
            v38 = *(_DWORD *)(v36 + 8);
            v39 = 1LL << v38;
            v40 = 8LL * (v38 >> 6);
LABEL_26:
            *(_QWORD *)(*(_QWORD *)(a1 + 2280) + v40) |= v39;
            v29 = v54;
            v41 = v53 + 1;
            v53 = v41;
            if ( v41 == v54 )
            {
LABEL_29:
              if ( (unsigned __int8 **)v57[0] == v54 )
                break;
            }
            else
            {
              while ( (unsigned __int64)(*v41 + 2) <= 1 )
              {
                v53 = ++v41;
                if ( v41 == v54 )
                  goto LABEL_29;
              }
              v29 = v53;
              if ( (unsigned __int8 **)v57[0] == v53 )
                break;
            }
          }
        }
        if ( !*((_BYTE *)v23 + 36) )
          _libc_free(v23[2]);
        *v23 = -8192;
        --*(_DWORD *)(a1 + 1904);
        ++*(_DWORD *)(a1 + 1908);
      }
    }
    else
    {
      v50 = 1;
      while ( v24 != -4096 )
      {
        v22 = (v20 - 1) & (v50 + v22);
        v23 = (__int64 *)(v21 + 56LL * v22);
        v24 = *v23;
        if ( *v23 == a2 )
          goto LABEL_16;
        ++v50;
      }
    }
  }
}
