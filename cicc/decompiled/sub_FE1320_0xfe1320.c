// Function: sub_FE1320
// Address: 0xfe1320
//
__int64 __fastcall sub_FE1320(_QWORD *a1, __int64 a2)
{
  __int64 v3; // r12
  unsigned int v4; // eax
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  unsigned int v7; // ebx
  unsigned int *v9; // r12
  __int64 v10; // r15
  unsigned __int64 v11; // rax
  char v12; // dl
  __int64 v13; // r14
  _DWORD *v14; // rbx
  _DWORD *v15; // r15
  unsigned int *v16; // r14
  unsigned int *v17; // rbx
  __int64 v18; // rsi
  __int64 result; // rax
  int v20; // eax
  unsigned int *v21; // rdx
  unsigned int v22; // ecx
  int v23; // r9d
  unsigned int *v24; // r8
  int v25; // eax
  __int64 v26; // r14
  __int64 *v27; // rbx
  __int64 v28; // rax
  _DWORD *v29; // rdi
  __int64 v30; // r15
  __int64 v31; // rax
  _QWORD *v32; // r14
  __int64 v33; // rax
  unsigned int *v34; // rbx
  unsigned int *v35; // r14
  int v36; // edi
  unsigned int v37; // esi
  int v38; // ecx
  unsigned int *v39; // rdx
  int v40; // ecx
  int v41; // edi
  unsigned int v42; // esi
  _QWORD *v43; // [rsp+20h] [rbp-E0h]
  int v44; // [rsp+28h] [rbp-D8h]
  int v45; // [rsp+2Ch] [rbp-D4h]
  unsigned __int64 v46; // [rsp+30h] [rbp-D0h]
  char v47; // [rsp+38h] [rbp-C8h]
  __int64 v48; // [rsp+50h] [rbp-B0h] BYREF
  _DWORD *v49; // [rsp+58h] [rbp-A8h]
  __int64 v50; // [rsp+60h] [rbp-A0h]
  __int64 v51; // [rsp+68h] [rbp-98h]
  _QWORD v52[2]; // [rsp+70h] [rbp-90h] BYREF
  _BYTE v53[64]; // [rsp+80h] [rbp-80h] BYREF
  __int64 v54; // [rsp+C0h] [rbp-40h]
  char v55; // [rsp+C8h] [rbp-38h]

  v3 = a2;
  v4 = *(_DWORD *)(a2 + 12);
  if ( v4 > 1 )
  {
    v55 = 0;
    v52[0] = v53;
    v52[1] = 0x400000000LL;
    v54 = 0;
    v48 = 1;
    v49 = 0;
    v50 = 0;
    v51 = 0;
    v5 = (4 * v4 / 3 + 1) | ((unsigned __int64)(4 * v4 / 3 + 1) >> 1);
    v6 = (((v5 >> 2) | v5) >> 4) | (v5 >> 2) | v5;
    sub_A08C50((__int64)&v48, ((((v6 >> 8) | v6) >> 16) | (v6 >> 8) | v6) + 1);
    v44 = *(_DWORD *)(a2 + 12);
    if ( !v44 )
    {
      v13 = 1;
      goto LABEL_13;
    }
    v47 = 0;
    v7 = 0;
    v43 = a1 + 4;
    v46 = 0;
    v45 = 0;
    v44 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v9 = (unsigned int *)(*(_QWORD *)(a2 + 96) + 4LL * v7);
        v10 = *(_QWORD *)(a1[17] + 8LL * *v9);
        sub_FDE240(v43, *v9);
        v11 = sub_AA5EE0(v10);
        if ( v12 )
          break;
        if ( !(_DWORD)v51 )
        {
          ++v48;
          goto LABEL_66;
        }
        v20 = (v51 - 1) & v45;
        v21 = &v49[v20];
        v22 = *v21;
        if ( *v21 != v7 )
        {
          v23 = 1;
          v24 = 0;
          while ( v22 != -1 )
          {
            if ( v22 == -2 && !v24 )
              v24 = v21;
            v20 = (v51 - 1) & (v23 + v20);
            v21 = &v49[v20];
            v22 = *v21;
            if ( *v21 == v7 )
              goto LABEL_4;
            ++v23;
          }
          if ( !v24 )
            v24 = v21;
          ++v48;
          v25 = v50 + 1;
          if ( 4 * ((int)v50 + 1) < (unsigned int)(3 * v51) )
          {
            if ( (int)v51 - HIDWORD(v50) - v25 <= (unsigned int)v51 >> 3 )
            {
              sub_A08C50((__int64)&v48, v51);
              if ( !(_DWORD)v51 )
              {
LABEL_91:
                LODWORD(v50) = v50 + 1;
                goto LABEL_92;
              }
              v40 = 1;
              v39 = 0;
              v41 = (v51 - 1) & v45;
              v24 = &v49[v41];
              v42 = *v24;
              v25 = v50 + 1;
              if ( v7 != *v24 )
              {
                while ( v42 != -1 )
                {
                  if ( v42 == -2 && !v39 )
                    v39 = v24;
                  v41 = (v51 - 1) & (v40 + v41);
                  v24 = &v49[v41];
                  v42 = *v24;
                  if ( *v24 == v7 )
                    goto LABEL_31;
                  ++v40;
                }
                goto LABEL_70;
              }
            }
            goto LABEL_31;
          }
LABEL_66:
          sub_A08C50((__int64)&v48, 2 * v51);
          if ( !(_DWORD)v51 )
            goto LABEL_91;
          v36 = (v51 - 1) & v45;
          v24 = &v49[v36];
          v37 = *v24;
          v25 = v50 + 1;
          if ( v7 != *v24 )
          {
            v38 = 1;
            v39 = 0;
            while ( v37 != -1 )
            {
              if ( !v39 && v37 == -2 )
                v39 = v24;
              v36 = (v51 - 1) & (v38 + v36);
              v24 = &v49[v36];
              v37 = *v24;
              if ( *v24 == v7 )
                goto LABEL_31;
              ++v38;
            }
LABEL_70:
            if ( v39 )
              v24 = v39;
          }
LABEL_31:
          LODWORD(v50) = v25;
          if ( *v24 != -1 )
            --HIDWORD(v50);
          *v24 = v7;
        }
LABEL_4:
        v45 += 37;
        if ( *(_DWORD *)(a2 + 12) <= ++v7 )
          goto LABEL_11;
      }
      ++v44;
      if ( !v47 || v11 < v46 )
      {
        v46 = v11;
        v47 = 1;
      }
      if ( !v11 )
        goto LABEL_4;
      ++v7;
      sub_FE8630(v52, v9, v11, 0);
      v45 += 37;
      if ( *(_DWORD *)(a2 + 12) <= v7 )
      {
LABEL_11:
        v3 = a2;
        v13 = v46;
        if ( !v47 )
          v13 = 1;
LABEL_13:
        v14 = v49;
        v15 = &v49[(unsigned int)v51];
        if ( (_DWORD)v50 && v15 != v49 )
        {
          while ( *v14 > 0xFFFFFFFD )
          {
            if ( v15 == ++v14 )
              goto LABEL_14;
          }
          if ( v15 != v14 )
          {
            if ( v13 )
              goto LABEL_60;
            while ( ++v14 != v15 )
            {
              while ( *v14 > 0xFFFFFFFD )
              {
                if ( v15 == ++v14 )
                  goto LABEL_14;
              }
              if ( v15 == v14 )
                break;
              if ( v13 )
LABEL_60:
                sub_FE8630(v52, *(_QWORD *)(v3 + 96) + 4LL * (unsigned int)*v14, v13, 0);
            }
          }
        }
LABEL_14:
        sub_FEAD50(a1, v52);
        v16 = *(unsigned int **)(v3 + 96);
        v17 = &v16[*(unsigned int *)(v3 + 104)];
        if ( v17 != v16 )
        {
          while ( (unsigned __int8)sub_FDE3B0(a1, (_QWORD *)v3, v16) )
          {
            if ( v17 == ++v16 )
              goto LABEL_17;
          }
LABEL_92:
          BUG();
        }
LABEL_17:
        if ( !v44 )
          sub_FEAA90(a1, v3);
        v18 = 4LL * (unsigned int)v51;
        sub_C7D6A0((__int64)v49, v18, 4);
        if ( (_BYTE *)v52[0] != v53 )
          _libc_free(v52[0], v18);
LABEL_21:
        sub_FE9590(a1, v3);
        sub_FE86B0(a1, v3);
        return 1;
      }
    }
  }
  v26 = a1[8] + 24LL * **(unsigned int **)(a2 + 96);
  v27 = *(__int64 **)(v26 + 8);
  if ( !v27 )
    goto LABEL_42;
  v28 = *((unsigned int *)v27 + 3);
  v29 = (_DWORD *)v27[12];
  if ( (unsigned int)v28 > 1 )
  {
    if ( !sub_FDC990(v29, &v29[v28], (_DWORD *)v26) )
      goto LABEL_42;
  }
  else if ( *(_DWORD *)v26 != *v29 )
  {
    goto LABEL_42;
  }
  if ( *((_BYTE *)v27 + 8) )
  {
    v30 = *v27;
    if ( !*v27
      || (v31 = *(unsigned int *)(v30 + 12), (unsigned int)v31 <= 1)
      || !sub_FDC990(*(_DWORD **)(v30 + 96), (_DWORD *)(*(_QWORD *)(v30 + 96) + 4 * v31), (_DWORD *)v26)
      || (v32 = (_QWORD *)(v30 + 152), !*(_BYTE *)(v30 + 8)) )
    {
      v32 = v27 + 19;
    }
    goto LABEL_43;
  }
LABEL_42:
  v32 = (_QWORD *)(v26 + 16);
LABEL_43:
  *v32 = -1;
  LODWORD(v52[0]) = **(_DWORD **)(a2 + 96);
  if ( !(unsigned __int8)sub_FDE3B0(a1, (_QWORD *)a2, (unsigned int *)v52) )
    goto LABEL_92;
  v33 = *(_QWORD *)(a2 + 96);
  v34 = (unsigned int *)(v33 + 4LL * *(unsigned int *)(a2 + 104));
  v35 = (unsigned int *)(v33 + 4LL * *(unsigned int *)(a2 + 12));
  if ( v34 == v35 )
    goto LABEL_21;
  while ( 1 )
  {
    result = sub_FDE3B0(a1, (_QWORD *)a2, v35);
    if ( !(_BYTE)result )
      return result;
    if ( v34 == ++v35 )
      goto LABEL_21;
  }
}
