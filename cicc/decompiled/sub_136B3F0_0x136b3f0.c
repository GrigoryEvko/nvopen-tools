// Function: sub_136B3F0
// Address: 0x136b3f0
//
__int64 __fastcall sub_136B3F0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r14
  unsigned int v5; // eax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned int v8; // ebx
  unsigned __int64 v9; // rax
  unsigned int *v10; // r15
  __int64 v11; // r14
  int v12; // edx
  unsigned int *v13; // rcx
  unsigned int v14; // edi
  int v15; // r8d
  unsigned int *v16; // rax
  int v17; // edx
  __int64 v18; // r14
  _DWORD *v19; // rbx
  _DWORD *v20; // r15
  unsigned int *v21; // r14
  unsigned int *v22; // rbx
  unsigned int *v23; // rdx
  __int64 result; // rax
  __int64 v25; // r14
  __int64 *v26; // rbx
  __int64 v27; // rax
  _DWORD *v28; // rdi
  __int64 v29; // r15
  __int64 v30; // rax
  _QWORD *v31; // r14
  __int64 v32; // rax
  unsigned int *v33; // rbx
  unsigned int *v34; // r14
  int v35; // r9d
  unsigned int v36; // edi
  int v37; // esi
  unsigned int *v38; // rcx
  int v39; // esi
  int v40; // r9d
  unsigned int v41; // edi
  __int64 *v42; // [rsp+20h] [rbp-E0h]
  int v43; // [rsp+28h] [rbp-D8h]
  int v44; // [rsp+2Ch] [rbp-D4h]
  unsigned __int64 v45; // [rsp+30h] [rbp-D0h]
  char v46; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v47; // [rsp+40h] [rbp-C0h] BYREF
  char v48; // [rsp+48h] [rbp-B8h]
  __int64 v49; // [rsp+50h] [rbp-B0h] BYREF
  _DWORD *v50; // [rsp+58h] [rbp-A8h]
  __int64 v51; // [rsp+60h] [rbp-A0h]
  __int64 v52; // [rsp+68h] [rbp-98h]
  unsigned __int64 v53[2]; // [rsp+70h] [rbp-90h] BYREF
  _BYTE v54[64]; // [rsp+80h] [rbp-80h] BYREF
  __int64 v55; // [rsp+C0h] [rbp-40h]
  char v56; // [rsp+C8h] [rbp-38h]

  v5 = *(_DWORD *)(a2 + 12);
  if ( v5 > 1 )
  {
    v56 = 0;
    v53[0] = (unsigned __int64)v54;
    v53[1] = 0x400000000LL;
    v55 = 0;
    v49 = 1;
    v50 = 0;
    v51 = 0;
    v52 = 0;
    v6 = (4 * v5 / 3 + 1) | ((unsigned __int64)(4 * v5 / 3 + 1) >> 1);
    v7 = (((v6 >> 2) | v6) >> 4) | (v6 >> 2) | v6;
    sub_136B240((__int64)&v49, ((((v7 >> 8) | v7) >> 16) | (v7 >> 8) | v7) + 1);
    v43 = *(_DWORD *)(a2 + 12);
    if ( !v43 )
    {
      v18 = 1;
      goto LABEL_25;
    }
    v8 = 0;
    v44 = 0;
    v42 = (__int64 *)(a1 + 32);
    v46 = 0;
    v43 = 0;
    v45 = v2;
    while ( 1 )
    {
      v10 = (unsigned int *)(*(_QWORD *)(a2 + 96) + 4LL * v8);
      v11 = *(_QWORD *)(*(_QWORD *)(a1 + 136) + 8LL * *v10);
      sub_1369D60(v42, *v10);
      sub_157F7D0(&v47, v11);
      if ( v48 )
      {
        ++v43;
        if ( v46 )
        {
          v9 = v45;
          if ( v45 > v47 )
            v9 = v47;
          v45 = v9;
          if ( !v47 )
            goto LABEL_7;
LABEL_11:
          sub_1370BE0(v53, v10, v47, 0);
          goto LABEL_7;
        }
        v45 = v47;
        v46 = 1;
        if ( v47 )
          goto LABEL_11;
      }
      else
      {
        if ( !(_DWORD)v52 )
        {
          ++v49;
          goto LABEL_65;
        }
        v12 = (v52 - 1) & v44;
        v13 = &v50[v12];
        v14 = *v13;
        if ( *v13 != v8 )
        {
          v15 = 1;
          v16 = 0;
          while ( v14 != -1 )
          {
            if ( v14 == -2 && !v16 )
              v16 = v13;
            v12 = (v52 - 1) & (v15 + v12);
            v13 = &v50[v12];
            v14 = *v13;
            if ( *v13 == v8 )
              goto LABEL_7;
            ++v15;
          }
          if ( !v16 )
            v16 = v13;
          ++v49;
          v17 = v51 + 1;
          if ( 4 * ((int)v51 + 1) < (unsigned int)(3 * v52) )
          {
            if ( (int)v52 - HIDWORD(v51) - v17 <= (unsigned int)v52 >> 3 )
            {
              sub_136B240((__int64)&v49, v52);
              if ( !(_DWORD)v52 )
              {
LABEL_90:
                LODWORD(v51) = v51 + 1;
                BUG();
              }
              v39 = 1;
              v40 = (v52 - 1) & v44;
              v16 = &v50[v40];
              v17 = v51 + 1;
              v38 = 0;
              v41 = *v16;
              if ( v8 != *v16 )
              {
                while ( v41 != -1 )
                {
                  if ( v41 == -2 && !v38 )
                    v38 = v16;
                  v40 = (v52 - 1) & (v39 + v40);
                  v16 = &v50[v40];
                  v41 = *v16;
                  if ( *v16 == v8 )
                    goto LABEL_20;
                  ++v39;
                }
                goto LABEL_69;
              }
            }
            goto LABEL_20;
          }
LABEL_65:
          sub_136B240((__int64)&v49, 2 * v52);
          if ( !(_DWORD)v52 )
            goto LABEL_90;
          v35 = (v52 - 1) & v44;
          v16 = &v50[v35];
          v17 = v51 + 1;
          v36 = *v16;
          if ( v8 != *v16 )
          {
            v37 = 1;
            v38 = 0;
            while ( v36 != -1 )
            {
              if ( !v38 && v36 == -2 )
                v38 = v16;
              v35 = (v52 - 1) & (v37 + v35);
              v16 = &v50[v35];
              v36 = *v16;
              if ( *v16 == v8 )
                goto LABEL_20;
              ++v37;
            }
LABEL_69:
            if ( v38 )
              v16 = v38;
          }
LABEL_20:
          LODWORD(v51) = v17;
          if ( *v16 != -1 )
            --HIDWORD(v51);
          *v16 = v8;
        }
      }
LABEL_7:
      v44 += 37;
      if ( *(_DWORD *)(a2 + 12) <= ++v8 )
      {
        v18 = v45;
        if ( !v46 )
          v18 = 1;
LABEL_25:
        v19 = v50;
        v20 = &v50[(unsigned int)v52];
        if ( (_DWORD)v51 && v20 != v50 )
        {
          while ( *v19 > 0xFFFFFFFD )
          {
            if ( v20 == ++v19 )
              goto LABEL_26;
          }
          if ( v20 != v19 )
          {
            if ( v18 )
              goto LABEL_59;
            while ( ++v19 != v20 )
            {
              while ( *v19 > 0xFFFFFFFD )
              {
                if ( v20 == ++v19 )
                  goto LABEL_26;
              }
              if ( v20 == v19 )
                break;
              if ( v18 )
LABEL_59:
                sub_1370BE0(v53, *(_QWORD *)(a2 + 96) + 4LL * (unsigned int)*v19, v18, 0);
            }
          }
        }
LABEL_26:
        sub_1373B30(a1, v53);
        v21 = *(unsigned int **)(a2 + 96);
        v22 = &v21[*(unsigned int *)(a2 + 104)];
        while ( v22 != v21 )
        {
          v23 = v21++;
          sub_1369ED0(a1, a2, v23);
        }
        if ( !v43 )
          sub_1373870(a1, a2);
        j___libc_free_0(v50);
        if ( (_BYTE *)v53[0] != v54 )
          _libc_free(v53[0]);
LABEL_32:
        sub_1371D60(a1, a2);
        sub_1370C60(a1, a2);
        return 1;
      }
    }
  }
  v25 = *(_QWORD *)(a1 + 64) + 24LL * **(unsigned int **)(a2 + 96);
  v26 = *(__int64 **)(v25 + 8);
  if ( !v26 )
    goto LABEL_42;
  v27 = *((unsigned int *)v26 + 3);
  v28 = (_DWORD *)v26[12];
  if ( (unsigned int)v27 > 1 )
  {
    if ( !sub_1369030(v28, &v28[v27], (_DWORD *)v25) )
      goto LABEL_42;
  }
  else if ( *(_DWORD *)v25 != *v28 )
  {
    goto LABEL_42;
  }
  if ( *((_BYTE *)v26 + 8) )
  {
    v29 = *v26;
    if ( !*v26
      || (v30 = *(unsigned int *)(v29 + 12), (unsigned int)v30 <= 1)
      || !sub_1369030(*(_DWORD **)(v29 + 96), (_DWORD *)(*(_QWORD *)(v29 + 96) + 4 * v30), (_DWORD *)v25)
      || (v31 = (_QWORD *)(v29 + 152), !*(_BYTE *)(v29 + 8)) )
    {
      v31 = v26 + 19;
    }
    goto LABEL_43;
  }
LABEL_42:
  v31 = (_QWORD *)(v25 + 16);
LABEL_43:
  *v31 = -1;
  LODWORD(v53[0]) = **(_DWORD **)(a2 + 96);
  sub_1369ED0(a1, a2, (unsigned int *)v53);
  v32 = *(_QWORD *)(a2 + 96);
  v33 = (unsigned int *)(v32 + 4LL * *(unsigned int *)(a2 + 104));
  v34 = (unsigned int *)(v32 + 4LL * *(unsigned int *)(a2 + 12));
  if ( v33 == v34 )
    goto LABEL_32;
  while ( 1 )
  {
    result = sub_1369ED0(a1, a2, v34);
    if ( !(_BYTE)result )
      return result;
    if ( v33 == ++v34 )
      goto LABEL_32;
  }
}
