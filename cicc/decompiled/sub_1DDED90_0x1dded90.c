// Function: sub_1DDED90
// Address: 0x1dded90
//
__int64 __fastcall sub_1DDED90(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r14
  unsigned int v5; // eax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned int v8; // ebx
  int v9; // edx
  unsigned int *v10; // rcx
  unsigned int v11; // edi
  unsigned int *v12; // r15
  __int64 v13; // r14
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // r14
  _DWORD *v17; // rbx
  _DWORD *v18; // r15
  unsigned int *v19; // r14
  unsigned int *v20; // rbx
  unsigned int *v21; // rdx
  __int64 result; // rax
  __int64 v23; // r14
  __int64 *v24; // rbx
  __int64 v25; // rax
  _DWORD *v26; // rdi
  __int64 v27; // r15
  __int64 v28; // rax
  _QWORD *v29; // r14
  __int64 v30; // rax
  unsigned int *v31; // rbx
  unsigned int *v32; // r14
  int v33; // r8d
  unsigned int *v34; // rax
  int v35; // edx
  int v36; // r9d
  unsigned int v37; // edi
  int v38; // esi
  unsigned int *v39; // rcx
  int v40; // esi
  int v41; // r9d
  unsigned int v42; // edi
  __int64 *v43; // [rsp+10h] [rbp-D0h]
  int v44; // [rsp+18h] [rbp-C8h]
  int v45; // [rsp+1Ch] [rbp-C4h]
  unsigned __int64 v46; // [rsp+20h] [rbp-C0h]
  char v47; // [rsp+28h] [rbp-B8h]
  __int64 v48; // [rsp+30h] [rbp-B0h] BYREF
  _DWORD *v49; // [rsp+38h] [rbp-A8h]
  __int64 v50; // [rsp+40h] [rbp-A0h]
  __int64 v51; // [rsp+48h] [rbp-98h]
  unsigned __int64 v52[2]; // [rsp+50h] [rbp-90h] BYREF
  _BYTE v53[64]; // [rsp+60h] [rbp-80h] BYREF
  __int64 v54; // [rsp+A0h] [rbp-40h]
  char v55; // [rsp+A8h] [rbp-38h]

  v5 = *(_DWORD *)(a2 + 12);
  if ( v5 > 1 )
  {
    v55 = 0;
    v52[1] = 0x400000000LL;
    v52[0] = (unsigned __int64)v53;
    v54 = 0;
    v48 = 1;
    v49 = 0;
    v50 = 0;
    v51 = 0;
    v6 = (4 * v5 / 3 + 1) | ((unsigned __int64)(4 * v5 / 3 + 1) >> 1);
    v7 = (((v6 >> 2) | v6) >> 4) | (v6 >> 2) | v6;
    sub_136B240((__int64)&v48, ((((v7 >> 8) | v7) >> 16) | (v7 >> 8) | v7) + 1);
    v44 = *(_DWORD *)(a2 + 12);
    if ( !v44 )
    {
      v16 = 1;
      goto LABEL_17;
    }
    v47 = 0;
    v8 = 0;
    v43 = (__int64 *)(a1 + 32);
    v45 = 0;
    v44 = 0;
    v46 = v2;
    while ( 1 )
    {
      v12 = (unsigned int *)(*(_QWORD *)(a2 + 96) + 4LL * v8);
      v13 = *(_QWORD *)(*(_QWORD *)(a1 + 136) + 8LL * *v12);
      sub_1369D60(v43, *v12);
      if ( *(_BYTE *)(v13 + 144) )
      {
        ++v44;
        v14 = *(_QWORD *)(v13 + 136);
        if ( v47 )
        {
          v15 = v46;
          if ( v46 > v14 )
            v15 = *(_QWORD *)(v13 + 136);
          v46 = v15;
        }
        else
        {
          v46 = *(_QWORD *)(v13 + 136);
          v47 = 1;
        }
        if ( v14 )
          sub_1370BE0((__int64)v52, v12, v14, 0);
      }
      else
      {
        if ( !(_DWORD)v51 )
        {
          ++v48;
          goto LABEL_66;
        }
        v9 = (v51 - 1) & v45;
        v10 = &v49[v9];
        v11 = *v10;
        if ( *v10 != v8 )
        {
          v33 = 1;
          v34 = 0;
          while ( v11 != -1 )
          {
            if ( v11 == -2 && !v34 )
              v34 = v10;
            v9 = (v51 - 1) & (v33 + v9);
            v10 = &v49[v9];
            v11 = *v10;
            if ( *v10 == v8 )
              goto LABEL_6;
            ++v33;
          }
          if ( !v34 )
            v34 = v10;
          ++v48;
          v35 = v50 + 1;
          if ( 4 * ((int)v50 + 1) < (unsigned int)(3 * v51) )
          {
            if ( (int)v51 - HIDWORD(v50) - v35 <= (unsigned int)v51 >> 3 )
            {
              sub_136B240((__int64)&v48, v51);
              if ( !(_DWORD)v51 )
              {
LABEL_91:
                LODWORD(v50) = v50 + 1;
                BUG();
              }
              v40 = 1;
              v41 = (v51 - 1) & v45;
              v34 = &v49[v41];
              v35 = v50 + 1;
              v39 = 0;
              v42 = *v34;
              if ( *v34 != v8 )
              {
                while ( v42 != -1 )
                {
                  if ( v42 == -2 && !v39 )
                    v39 = v34;
                  v41 = (v51 - 1) & (v40 + v41);
                  v34 = &v49[v41];
                  v42 = *v34;
                  if ( *v34 == v8 )
                    goto LABEL_59;
                  ++v40;
                }
                goto LABEL_70;
              }
            }
            goto LABEL_59;
          }
LABEL_66:
          sub_136B240((__int64)&v48, 2 * v51);
          if ( !(_DWORD)v51 )
            goto LABEL_91;
          v36 = (v51 - 1) & v45;
          v34 = &v49[v36];
          v35 = v50 + 1;
          v37 = *v34;
          if ( *v34 != v8 )
          {
            v38 = 1;
            v39 = 0;
            while ( v37 != -1 )
            {
              if ( !v39 && v37 == -2 )
                v39 = v34;
              v36 = (v51 - 1) & (v38 + v36);
              v34 = &v49[v36];
              v37 = *v34;
              if ( *v34 == v8 )
                goto LABEL_59;
              ++v38;
            }
LABEL_70:
            if ( v39 )
              v34 = v39;
          }
LABEL_59:
          LODWORD(v50) = v35;
          if ( *v34 != -1 )
            --HIDWORD(v50);
          *v34 = v8;
        }
      }
LABEL_6:
      v45 += 37;
      if ( *(_DWORD *)(a2 + 12) <= ++v8 )
      {
        v16 = v46;
        if ( !v47 )
          v16 = 1;
LABEL_17:
        v17 = v49;
        v18 = &v49[(unsigned int)v51];
        if ( (_DWORD)v50 && v49 != v18 )
        {
          while ( *v17 > 0xFFFFFFFD )
          {
            if ( v18 == ++v17 )
              goto LABEL_18;
          }
          if ( v17 != v18 )
          {
            if ( v16 )
              goto LABEL_51;
            while ( ++v17 != v18 )
            {
              while ( *v17 > 0xFFFFFFFD )
              {
                if ( v18 == ++v17 )
                  goto LABEL_18;
              }
              if ( v17 == v18 )
                break;
              if ( v16 )
LABEL_51:
                sub_1370BE0((__int64)v52, (unsigned int *)(*(_QWORD *)(a2 + 96) + 4LL * (unsigned int)*v17), v16, 0);
            }
          }
        }
LABEL_18:
        sub_1373B30(a1, (__int64)v52);
        v19 = *(unsigned int **)(a2 + 96);
        v20 = &v19[*(unsigned int *)(a2 + 104)];
        while ( v20 != v19 )
        {
          v21 = v19++;
          sub_1DDCDE0(a1, a2, v21);
        }
        if ( !v44 )
          sub_1373870(a1, a2);
        j___libc_free_0(v49);
        if ( (_BYTE *)v52[0] != v53 )
          _libc_free(v52[0]);
LABEL_24:
        sub_1371D60(a1, a2);
        sub_1370C60(a1, a2);
        return 1;
      }
    }
  }
  v23 = *(_QWORD *)(a1 + 64) + 24LL * **(unsigned int **)(a2 + 96);
  v24 = *(__int64 **)(v23 + 8);
  if ( !v24 )
    goto LABEL_34;
  v25 = *((unsigned int *)v24 + 3);
  v26 = (_DWORD *)v24[12];
  if ( (unsigned int)v25 > 1 )
  {
    if ( !sub_1369030(v26, &v26[v25], (_DWORD *)v23) )
      goto LABEL_34;
  }
  else if ( *(_DWORD *)v23 != *v26 )
  {
    goto LABEL_34;
  }
  if ( *((_BYTE *)v24 + 8) )
  {
    v27 = *v24;
    if ( !*v24
      || (v28 = *(unsigned int *)(v27 + 12), (unsigned int)v28 <= 1)
      || !sub_1369030(*(_DWORD **)(v27 + 96), (_DWORD *)(*(_QWORD *)(v27 + 96) + 4 * v28), (_DWORD *)v23)
      || (v29 = (_QWORD *)(v27 + 152), !*(_BYTE *)(v27 + 8)) )
    {
      v29 = v24 + 19;
    }
    goto LABEL_35;
  }
LABEL_34:
  v29 = (_QWORD *)(v23 + 16);
LABEL_35:
  *v29 = -1;
  LODWORD(v52[0]) = **(_DWORD **)(a2 + 96);
  sub_1DDCDE0(a1, a2, (unsigned int *)v52);
  v30 = *(_QWORD *)(a2 + 96);
  v31 = (unsigned int *)(v30 + 4LL * *(unsigned int *)(a2 + 104));
  v32 = (unsigned int *)(v30 + 4LL * *(unsigned int *)(a2 + 12));
  if ( v31 == v32 )
    goto LABEL_24;
  while ( 1 )
  {
    result = sub_1DDCDE0(a1, a2, v32);
    if ( !(_BYTE)result )
      return result;
    if ( v31 == ++v32 )
      goto LABEL_24;
  }
}
