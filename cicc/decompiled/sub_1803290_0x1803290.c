// Function: sub_1803290
// Address: 0x1803290
//
char __fastcall sub_1803290(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned int v4; // r13d
  int v5; // eax
  __int64 v6; // r8
  int v7; // eax
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rdi
  unsigned int v11; // ebx
  bool v12; // al
  __int64 v13; // rdi
  unsigned int v14; // ebx
  __int64 v15; // r15
  __int64 v16; // rdx
  __int64 v17; // rbx
  unsigned int v18; // r8d
  __int64 v19; // r10
  unsigned int v20; // r9d
  __int64 *v21; // r12
  __int64 v22; // rdi
  __int64 v23; // rbx
  __int64 v24; // rdx
  __int64 v25; // r9
  unsigned int v26; // edx
  unsigned int v27; // edx
  int v28; // esi
  __int64 *v29; // rcx
  int v30; // eax
  int v31; // eax
  int v32; // edi
  int v33; // edi
  __int64 v34; // r9
  unsigned int v35; // edx
  __int64 v36; // r8
  int v37; // ecx
  __int64 *v38; // r10
  int v39; // edi
  int v40; // edi
  __int64 v41; // r9
  int v42; // ecx
  int v43; // edx
  __int64 v44; // r8
  int v45; // r11d
  __int64 v46; // rcx
  int v47; // r11d
  __int64 v48; // rcx
  __int64 v50; // [rsp+8h] [rbp-78h]
  __int64 v51; // [rsp+10h] [rbp-70h]
  __int64 v52; // [rsp+10h] [rbp-70h]
  int v53; // [rsp+18h] [rbp-68h]
  int v54; // [rsp+18h] [rbp-68h]
  int v55; // [rsp+1Ch] [rbp-64h]
  __int64 v56; // [rsp+20h] [rbp-60h]
  unsigned int v57; // [rsp+28h] [rbp-58h]
  const char *v58; // [rsp+30h] [rbp-50h] BYREF
  char v59; // [rsp+40h] [rbp-40h]
  char v60; // [rsp+41h] [rbp-3Fh]

  *(_BYTE *)a1 = 1;
  v60 = 1;
  v58 = "llvm.asan.globals";
  v59 = 3;
  v3 = sub_1632310(a2, (__int64)&v58);
  v56 = v3;
  if ( v3 )
  {
    LODWORD(v3) = sub_161F520(v3);
    v55 = v3;
    if ( (_DWORD)v3 )
    {
      v4 = 0;
      v50 = a1 + 8;
      while ( 1 )
      {
        v15 = sub_161F530(v56, v4);
        v3 = *(unsigned int *)(v15 + 8);
        v16 = *(_QWORD *)(v15 - 8 * v3);
        if ( v16 )
        {
          v17 = *(_QWORD *)(v16 + 136);
          if ( v17 )
            break;
        }
LABEL_15:
        if ( v55 == ++v4 )
          return v3;
      }
      v18 = *(_DWORD *)(a1 + 32);
      if ( v18 )
      {
        v19 = *(_QWORD *)(a1 + 16);
        v20 = (v18 - 1) & (((unsigned int)v17 >> 4) ^ ((unsigned int)v17 >> 9));
        v21 = (__int64 *)(v19 + 56LL * v20);
        v22 = *v21;
        if ( v17 == *v21 )
        {
LABEL_20:
          v23 = *(_QWORD *)(v15 + 8 * (1 - v3));
          if ( v23 )
          {
            v21[1] = sub_161E970(*(_QWORD *)(v23 - 8LL * *(unsigned int *)(v23 + 8)));
            v21[2] = v24;
            v25 = *(_QWORD *)(*(_QWORD *)(v23 + 8 * (1LL - *(unsigned int *)(v23 + 8))) + 136LL);
            if ( *(_DWORD *)(v25 + 32) <= 0x40u )
            {
              v5 = *(_DWORD *)(v25 + 24);
            }
            else
            {
              v51 = *(_QWORD *)(*(_QWORD *)(v23 + 8 * (1LL - *(unsigned int *)(v23 + 8))) + 136LL);
              v54 = *(_DWORD *)(v25 + 32);
              v26 = v54 - sub_16A57B0(v25 + 24);
              v5 = -1;
              if ( v26 <= 0x40 )
                v5 = **(_DWORD **)(v51 + 24);
            }
            *((_DWORD *)v21 + 6) = v5;
            v6 = *(_QWORD *)(*(_QWORD *)(v23 + 8 * (2LL - *(unsigned int *)(v23 + 8))) + 136LL);
            if ( *(_DWORD *)(v6 + 32) > 0x40u )
            {
              v52 = *(_QWORD *)(*(_QWORD *)(v23 + 8 * (2LL - *(unsigned int *)(v23 + 8))) + 136LL);
              v53 = *(_DWORD *)(v6 + 32);
              v27 = v53 - sub_16A57B0(v6 + 24);
              v7 = -1;
              if ( v27 <= 0x40 )
                v7 = **(_DWORD **)(v52 + 24);
            }
            else
            {
              v7 = *(_DWORD *)(v6 + 24);
            }
            *((_DWORD *)v21 + 7) = v7;
            v3 = *(unsigned int *)(v15 + 8);
          }
          v8 = *(_QWORD *)(v15 + 8 * (2 - v3));
          if ( v8 )
          {
            v21[4] = sub_161E970(v8);
            v21[5] = v9;
            v3 = *(unsigned int *)(v15 + 8);
          }
          v10 = *(_QWORD *)(*(_QWORD *)(v15 + 8 * (3 - v3)) + 136LL);
          v11 = *(_DWORD *)(v10 + 32);
          if ( v11 <= 0x40 )
            v12 = *(_QWORD *)(v10 + 24) == 1;
          else
            v12 = v11 - 1 == (unsigned int)sub_16A57B0(v10 + 24);
          *((_BYTE *)v21 + 48) |= v12;
          v13 = *(_QWORD *)(*(_QWORD *)(v15 + 8 * (4LL - *(unsigned int *)(v15 + 8))) + 136LL);
          v14 = *(_DWORD *)(v13 + 32);
          if ( v14 <= 0x40 )
            LOBYTE(v3) = *(_QWORD *)(v13 + 24) == 1;
          else
            LOBYTE(v3) = v14 - 1 == (unsigned int)sub_16A57B0(v13 + 24);
          *((_BYTE *)v21 + 49) |= v3;
          goto LABEL_15;
        }
        v28 = 1;
        v29 = 0;
        while ( v22 != -8 )
        {
          if ( v22 == -16 && !v29 )
            v29 = v21;
          v20 = (v18 - 1) & (v28 + v20);
          v21 = (__int64 *)(v19 + 56LL * v20);
          v22 = *v21;
          if ( v17 == *v21 )
            goto LABEL_20;
          ++v28;
        }
        v30 = *(_DWORD *)(a1 + 24);
        if ( v29 )
          v21 = v29;
        ++*(_QWORD *)(a1 + 8);
        v31 = v30 + 1;
        if ( 4 * v31 < 3 * v18 )
        {
          if ( v18 - *(_DWORD *)(a1 + 28) - v31 > v18 >> 3 )
            goto LABEL_35;
          v57 = ((unsigned int)v17 >> 4) ^ ((unsigned int)v17 >> 9);
          sub_1801F60(v50, v18);
          v39 = *(_DWORD *)(a1 + 32);
          if ( !v39 )
          {
LABEL_64:
            ++*(_DWORD *)(a1 + 24);
            BUG();
          }
          v40 = v39 - 1;
          v41 = *(_QWORD *)(a1 + 16);
          v42 = 1;
          v38 = 0;
          v43 = v40 & v57;
          v21 = (__int64 *)(v41 + 56LL * (v40 & v57));
          v44 = *v21;
          v31 = *(_DWORD *)(a1 + 24) + 1;
          if ( v17 == *v21 )
            goto LABEL_35;
          while ( v44 != -8 )
          {
            if ( v44 == -16 && !v38 )
              v38 = v21;
            v45 = v42 + 1;
            v46 = v40 & (unsigned int)(v43 + v42);
            v43 = v46;
            v21 = (__int64 *)(v41 + 56 * v46);
            v44 = *v21;
            if ( v17 == *v21 )
              goto LABEL_35;
            v42 = v45;
          }
          goto LABEL_43;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 8);
      }
      sub_1801F60(v50, 2 * v18);
      v32 = *(_DWORD *)(a1 + 32);
      if ( !v32 )
        goto LABEL_64;
      v33 = v32 - 1;
      v34 = *(_QWORD *)(a1 + 16);
      v35 = v33 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v21 = (__int64 *)(v34 + 56LL * v35);
      v36 = *v21;
      v31 = *(_DWORD *)(a1 + 24) + 1;
      if ( v17 == *v21 )
        goto LABEL_35;
      v37 = 1;
      v38 = 0;
      while ( v36 != -8 )
      {
        if ( !v38 && v36 == -16 )
          v38 = v21;
        v47 = v37 + 1;
        v48 = v33 & (v35 + v37);
        v35 = v48;
        v21 = (__int64 *)(v34 + 56 * v48);
        v36 = *v21;
        if ( v17 == *v21 )
          goto LABEL_35;
        v37 = v47;
      }
LABEL_43:
      if ( v38 )
        v21 = v38;
LABEL_35:
      *(_DWORD *)(a1 + 24) = v31;
      if ( *v21 != -8 )
        --*(_DWORD *)(a1 + 28);
      *v21 = v17;
      *(_OWORD *)(v21 + 1) = 0;
      *(_OWORD *)(v21 + 3) = 0;
      *(_OWORD *)(v21 + 5) = 0;
      v3 = *(unsigned int *)(v15 + 8);
      goto LABEL_20;
    }
  }
  return v3;
}
