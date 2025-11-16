// Function: sub_CF8780
// Address: 0xcf8780
//
__int64 __fastcall sub_CF8780(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // r11
  void **v6; // rax
  void **v7; // rdx
  __int64 *v9; // rax
  __int64 *v10; // r13
  __int64 *v11; // r8
  __int64 v12; // rdi
  int v13; // esi
  unsigned int v14; // edx
  __int64 v15; // rax
  __int64 v16; // r9
  __int64 v17; // rdx
  char v18; // r14
  __int64 v19; // rbx
  __int64 v20; // r12
  char v21; // cl
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rsi
  int v26; // r10d
  unsigned int i; // eax
  _QWORD *v28; // rcx
  unsigned int v29; // eax
  __int64 v30; // rsi
  int v31; // eax
  char v32; // al
  char v33; // cl
  __int64 v34; // r9
  unsigned int v35; // edx
  __int64 v36; // rdi
  int v37; // esi
  __int64 v38; // r10
  unsigned int v39; // eax
  int v40; // edx
  unsigned int v41; // edi
  unsigned int v42; // esi
  int v43; // r10d
  __int64 v44; // rsi
  int v45; // edx
  unsigned int v46; // eax
  __int64 v47; // rcx
  __int64 v48; // rsi
  int v49; // edx
  unsigned int v50; // eax
  __int64 v51; // rcx
  int v52; // edi
  __int64 v53; // r9
  int v54; // edx
  int v55; // edx
  int v56; // edi
  __int64 v57; // [rsp+0h] [rbp-50h]
  __int64 *v58; // [rsp+8h] [rbp-48h]
  __int64 v59; // [rsp+8h] [rbp-48h]
  __int64 v60; // [rsp+8h] [rbp-48h]
  __int64 v62; // [rsp+10h] [rbp-40h]
  int v63; // [rsp+10h] [rbp-40h]
  __int64 *v64; // [rsp+10h] [rbp-40h]
  __int64 *v65; // [rsp+10h] [rbp-40h]

  v4 = a3;
  if ( *(_BYTE *)(a3 + 76) )
  {
    v6 = *(void ***)(a3 + 56);
    v7 = &v6[*(unsigned int *)(a3 + 68)];
    if ( v6 != v7 )
    {
      while ( *v6 != &unk_4F86540 )
      {
        if ( v7 == ++v6 )
          goto LABEL_8;
      }
      return 1;
    }
  }
  else
  {
    v9 = sub_C8CA60(a3 + 48, (__int64)&unk_4F86540);
    v4 = a3;
    if ( v9 )
      return 1;
  }
LABEL_8:
  v10 = *(__int64 **)(a1 + 32);
  v11 = *(__int64 **)(a1 + 40);
  if ( v11 != v10 )
  {
    v57 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    do
    {
      v19 = *a4;
      v20 = *v10;
      v21 = *(_BYTE *)(*a4 + 8) & 1;
      if ( v21 )
      {
        v12 = v19 + 16;
        v13 = 7;
      }
      else
      {
        v22 = *(unsigned int *)(v19 + 24);
        v12 = *(_QWORD *)(v19 + 16);
        if ( !(_DWORD)v22 )
          goto LABEL_27;
        v13 = v22 - 1;
      }
      v14 = v13 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v15 = v12 + 16LL * v14;
      v16 = *(_QWORD *)v15;
      if ( v20 == *(_QWORD *)v15 )
        goto LABEL_12;
      v31 = 1;
      while ( v16 != -4096 )
      {
        v43 = v31 + 1;
        v14 = v13 & (v31 + v14);
        v15 = v12 + 16LL * v14;
        v16 = *(_QWORD *)v15;
        if ( v20 == *(_QWORD *)v15 )
          goto LABEL_12;
        v31 = v43;
      }
      if ( v21 )
      {
        v30 = 128;
        goto LABEL_28;
      }
      v22 = *(unsigned int *)(v19 + 24);
LABEL_27:
      v30 = 16 * v22;
LABEL_28:
      v15 = v12 + v30;
LABEL_12:
      v17 = 128;
      if ( !v21 )
        v17 = 16LL * *(unsigned int *)(v19 + 24);
      if ( v15 == v12 + v17 )
      {
        v23 = a4[1];
        v24 = *(unsigned int *)(v23 + 24);
        v25 = *(_QWORD *)(v23 + 8);
        if ( (_DWORD)v24 )
        {
          v26 = 1;
          for ( i = (v24 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * (v57 | ((unsigned __int64)(((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4)) << 32))) >> 31)
                   ^ (484763065 * v57)); ; i = (v24 - 1) & v29 )
          {
            v28 = (_QWORD *)(v25 + 24LL * i);
            if ( v20 == *v28 && a2 == v28[1] )
              break;
            if ( *v28 == -4096 && v28[1] == -4096 )
              goto LABEL_35;
            v29 = v26 + i;
            ++v26;
          }
        }
        else
        {
LABEL_35:
          v28 = (_QWORD *)(v25 + 24 * v24);
        }
        v58 = v11;
        v62 = v4;
        v32 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64 *))(**(_QWORD **)(v28[2] + 24LL) + 16LL))(
                *(_QWORD *)(v28[2] + 24LL),
                a2,
                v4,
                a4);
        v4 = v62;
        v11 = v58;
        v18 = v32;
        v33 = *(_BYTE *)(v19 + 8) & 1;
        if ( v33 )
        {
          v34 = v19 + 16;
          v35 = ((unsigned __int8)((unsigned int)v20 >> 9) ^ (unsigned __int8)((unsigned int)v20 >> 4)) & 7;
          v15 = v19
              + 16
              + 16LL * (((unsigned __int8)((unsigned int)v20 >> 9) ^ (unsigned __int8)((unsigned int)v20 >> 4)) & 7);
          v36 = *(_QWORD *)v15;
          if ( v20 != *(_QWORD *)v15 )
          {
            v37 = 7;
LABEL_39:
            v63 = 1;
            v38 = 0;
            while ( v36 != -4096 )
            {
              if ( v36 == -8192 && !v38 )
                v38 = v15;
              v35 = v37 & (v63 + v35);
              v15 = v34 + 16LL * v35;
              v36 = *(_QWORD *)v15;
              if ( v20 == *(_QWORD *)v15 )
                goto LABEL_15;
              ++v63;
            }
            if ( !v38 )
              v38 = v15;
            v39 = *(_DWORD *)(v19 + 8);
            ++*(_QWORD *)v19;
            v40 = (v39 >> 1) + 1;
            if ( v33 )
            {
              v41 = 24;
              v42 = 8;
            }
            else
            {
              v42 = *(_DWORD *)(v19 + 24);
LABEL_51:
              v41 = 3 * v42;
            }
            if ( 4 * v40 >= v41 )
            {
              v59 = v4;
              v64 = v11;
              sub_BBCB10(v19, 2 * v42);
              v11 = v64;
              v4 = v59;
              if ( (*(_BYTE *)(v19 + 8) & 1) != 0 )
              {
                v44 = v19 + 16;
                v45 = 7;
              }
              else
              {
                v54 = *(_DWORD *)(v19 + 24);
                v44 = *(_QWORD *)(v19 + 16);
                if ( !v54 )
                  goto LABEL_92;
                v45 = v54 - 1;
              }
              v46 = v45 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
              v38 = v44 + 16LL * v46;
              v47 = *(_QWORD *)v38;
              if ( v20 != *(_QWORD *)v38 )
              {
                v56 = 1;
                v53 = 0;
                while ( v47 != -4096 )
                {
                  if ( !v53 && v47 == -8192 )
                    v53 = v38;
                  v46 = v45 & (v56 + v46);
                  v38 = v44 + 16LL * v46;
                  v47 = *(_QWORD *)v38;
                  if ( v20 == *(_QWORD *)v38 )
                    goto LABEL_62;
                  ++v56;
                }
LABEL_68:
                if ( v53 )
                  v38 = v53;
              }
            }
            else
            {
              if ( v42 - *(_DWORD *)(v19 + 12) - v40 > v42 >> 3 )
              {
LABEL_54:
                *(_DWORD *)(v19 + 8) = (2 * (v39 >> 1) + 2) | v39 & 1;
                if ( *(_QWORD *)v38 != -4096 )
                  --*(_DWORD *)(v19 + 12);
                *(_QWORD *)v38 = v20;
                *(_BYTE *)(v38 + 8) = v18;
                goto LABEL_16;
              }
              v60 = v4;
              v65 = v11;
              sub_BBCB10(v19, v42);
              v11 = v65;
              v4 = v60;
              if ( (*(_BYTE *)(v19 + 8) & 1) != 0 )
              {
                v48 = v19 + 16;
                v49 = 7;
              }
              else
              {
                v55 = *(_DWORD *)(v19 + 24);
                v48 = *(_QWORD *)(v19 + 16);
                if ( !v55 )
                {
LABEL_92:
                  *(_DWORD *)(v19 + 8) = (2 * (*(_DWORD *)(v19 + 8) >> 1) + 2) | *(_DWORD *)(v19 + 8) & 1;
                  BUG();
                }
                v49 = v55 - 1;
              }
              v50 = v49 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
              v38 = v48 + 16LL * v50;
              v51 = *(_QWORD *)v38;
              if ( v20 != *(_QWORD *)v38 )
              {
                v52 = 1;
                v53 = 0;
                while ( v51 != -4096 )
                {
                  if ( v51 == -8192 && !v53 )
                    v53 = v38;
                  v50 = v49 & (v52 + v50);
                  v38 = v48 + 16LL * v50;
                  v51 = *(_QWORD *)v38;
                  if ( v20 == *(_QWORD *)v38 )
                    goto LABEL_62;
                  ++v52;
                }
                goto LABEL_68;
              }
            }
LABEL_62:
            v39 = *(_DWORD *)(v19 + 8);
            goto LABEL_54;
          }
        }
        else
        {
          v42 = *(_DWORD *)(v19 + 24);
          if ( !v42 )
          {
            v39 = *(_DWORD *)(v19 + 8);
            ++*(_QWORD *)v19;
            v38 = 0;
            v40 = (v39 >> 1) + 1;
            goto LABEL_51;
          }
          v37 = v42 - 1;
          v34 = *(_QWORD *)(v19 + 16);
          v35 = v37 & (((unsigned int)v20 >> 4) ^ ((unsigned int)v20 >> 9));
          v15 = v34 + 16LL * v35;
          v36 = *(_QWORD *)v15;
          if ( v20 != *(_QWORD *)v15 )
            goto LABEL_39;
        }
      }
LABEL_15:
      v18 = *(_BYTE *)(v15 + 8);
LABEL_16:
      if ( v18 )
        return 1;
      ++v10;
    }
    while ( v11 != v10 );
  }
  return 0;
}
