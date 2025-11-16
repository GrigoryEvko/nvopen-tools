// Function: sub_1606860
// Address: 0x1606860
//
void __fastcall sub_1606860(__int64 a1, __int64 a2)
{
  int v2; // ebx
  __int64 v3; // r13
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 *v8; // rax
  __int64 *v9; // r15
  __int64 *v10; // r13
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 *v14; // rax
  unsigned int v15; // eax
  int v16; // r14d
  unsigned int v17; // ebx
  unsigned int v18; // eax
  __int64 v19; // r13
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rbx
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r13
  __int64 i; // r12
  __int64 v28; // rdi
  char v29; // al
  __int64 v30; // rcx
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // r12
  __int64 v34; // rsi
  __int64 v35; // rbx
  __int64 v36; // r13
  __int64 v37; // rsi
  __int64 v38; // r12
  __int64 v39; // rcx
  __int64 v40; // rsi
  __int64 v41; // rax
  __int64 v42; // r14
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 v46; // r13
  __int64 v47; // rbx
  __int64 j; // r12
  __int64 v49; // rdi
  __int64 v50; // rbx
  __int64 v51; // r12
  __int64 v52; // r12
  __int64 v53; // rbx
  __int64 v54; // [rsp+0h] [rbp-90h]
  __int64 v55; // [rsp+8h] [rbp-88h]
  __int64 v56; // [rsp+8h] [rbp-88h]
  __int64 v57; // [rsp+10h] [rbp-80h]
  __int64 v58; // [rsp+10h] [rbp-80h]
  __int64 v59; // [rsp+10h] [rbp-80h]
  __int64 v60; // [rsp+10h] [rbp-80h]
  __int64 v61; // [rsp+10h] [rbp-80h]
  __int64 v62; // [rsp+28h] [rbp-68h] BYREF
  __int64 v63; // [rsp+30h] [rbp-60h]
  __int64 v64; // [rsp+48h] [rbp-48h] BYREF
  __int64 v65; // [rsp+50h] [rbp-40h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 20) )
      return;
    if ( *(_DWORD *)(a1 + 24) <= 0x40u )
      goto LABEL_4;
    sub_1606530(a1, a2);
    if ( *(_DWORD *)(a1 + 24) )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 8));
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return;
    }
    goto LABEL_76;
  }
  v15 = 4 * v2;
  if ( (unsigned int)(4 * v2) < 0x40 )
    v15 = 64;
  if ( *(_DWORD *)(a1 + 24) > v15 )
  {
    v16 = 64;
    sub_1606530(a1, a2);
    v17 = v2 - 1;
    if ( v17 )
    {
      _BitScanReverse(&v18, v17);
      v16 = 1 << (33 - (v18 ^ 0x1F));
      if ( v16 < 64 )
        v16 = 64;
    }
    v19 = sub_16982B0(a1, a2);
    v22 = sub_16982C0(a1, a2, v20, v21);
    if ( *(_DWORD *)(a1 + 24) != v16 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 8));
      v23 = ((((((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
               | (4 * v16 / 3u + 1)
               | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
             | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
             | (4 * v16 / 3u + 1)
             | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
             | (4 * v16 / 3u + 1)
             | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
           | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
           | (4 * v16 / 3u + 1)
           | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 16;
      v24 = (v23
           | (((((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
               | (4 * v16 / 3u + 1)
               | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
             | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
             | (4 * v16 / 3u + 1)
             | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
             | (4 * v16 / 3u + 1)
             | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
           | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
           | (4 * v16 / 3u + 1)
           | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 24) = v24;
      v25 = sub_22077B0(40 * v24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = v25;
      if ( v19 == v22 )
        sub_169C630(&v64, v22, 1);
      else
        sub_1699170(&v64, v19, 1);
      v26 = *(_QWORD *)(a1 + 8);
      for ( i = v26 + 40LL * *(unsigned int *)(a1 + 24); i != v26; v26 += 40 )
      {
        if ( v26 )
        {
          v28 = v26 + 8;
          if ( v64 == v22 )
            sub_169C6E0(v28, &v64);
          else
            sub_16986C0(v28, &v64);
        }
      }
      if ( v64 == v22 )
      {
        v50 = v65;
        if ( v65 )
        {
          v51 = v65 + 32LL * *(_QWORD *)(v65 - 8);
          if ( v65 != v51 )
          {
            do
            {
              v51 -= 32;
              sub_127D120((_QWORD *)(v51 + 8));
            }
            while ( v50 != v51 );
          }
          j_j_j___libc_free_0_0(v50 - 8);
        }
        return;
      }
LABEL_86:
      sub_1698460(&v64);
      return;
    }
LABEL_76:
    *(_QWORD *)(a1 + 16) = 0;
    v42 = sub_16982B0(a1, a2);
    v45 = sub_16982C0(a1, a2, v43, v44);
    v46 = v45;
    if ( v42 == v45 )
      sub_169C630(&v64, v45, 1);
    else
      sub_1699170(&v64, v42, 1);
    v47 = *(_QWORD *)(a1 + 8);
    for ( j = v47 + 40LL * *(unsigned int *)(a1 + 24); j != v47; v47 += 40 )
    {
      if ( v47 )
      {
        v49 = v47 + 8;
        if ( v46 == v64 )
          sub_169C6E0(v49, &v64);
        else
          sub_16986C0(v49, &v64);
      }
    }
    if ( v46 == v64 )
    {
      v52 = v65;
      if ( v65 )
      {
        v53 = v65 + 32LL * *(_QWORD *)(v65 - 8);
        if ( v65 != v53 )
        {
          do
          {
            v53 -= 32;
            sub_127D120((_QWORD *)(v53 + 8));
          }
          while ( v52 != v53 );
        }
        j_j_j___libc_free_0_0(v52 - 8);
      }
      return;
    }
    goto LABEL_86;
  }
LABEL_4:
  v3 = sub_16982B0(a1, a2);
  v6 = sub_16982C0(a1, a2, v4, v5);
  v7 = v6;
  if ( v3 == v6 )
  {
    sub_169C630(&v62, v6, 1);
    sub_169C630(&v64, v7, 2);
  }
  else
  {
    sub_1699170(&v62, v3, 1);
    sub_1699170(&v64, v3, 2);
  }
  v8 = *(__int64 **)(a1 + 8);
  v9 = v8 + 1;
  v10 = &v8[5 * *(unsigned int *)(a1 + 24)];
  if ( v10 != v8 )
  {
    while ( 1 )
    {
      v11 = *v9;
      if ( *v9 == v62 )
      {
        if ( v7 == v11 )
        {
          if ( (unsigned __int8)sub_169CB90(v9, &v62) )
            goto LABEL_39;
        }
        else if ( (unsigned __int8)sub_1698510(v9, &v62) )
        {
          goto LABEL_39;
        }
        v11 = *v9;
      }
      if ( v64 == v11 && (v7 == v11 ? (v29 = sub_169CB90(v9, &v64)) : (v29 = sub_1698510(v9, &v64)), v29) )
      {
        v13 = v62;
        if ( v7 == *v9 )
          goto LABEL_45;
      }
      else
      {
        v12 = v9[3];
        if ( v12 )
        {
          if ( v7 == *(_QWORD *)(v12 + 32) )
          {
            v39 = *(_QWORD *)(v12 + 40);
            if ( v39 )
            {
              v40 = 32LL * *(_QWORD *)(v39 - 8);
              v41 = v39 + v40;
              if ( v39 != v39 + v40 )
              {
                do
                {
                  v54 = v39;
                  v56 = v12;
                  v60 = v41 - 32;
                  sub_127D120((_QWORD *)(v41 - 24));
                  v41 = v60;
                  v39 = v54;
                  v12 = v56;
                }
                while ( v54 != v60 );
              }
              v61 = v12;
              j_j_j___libc_free_0_0(v39 - 8);
              v12 = v61;
            }
          }
          else
          {
            v57 = v9[3];
            sub_1698460(v12 + 32);
            v12 = v57;
          }
          v58 = v12;
          sub_164BE60(v12);
          sub_1648B90(v58);
        }
        v13 = v62;
        if ( v7 == *v9 )
        {
LABEL_45:
          if ( v7 == v13 )
          {
            sub_16A0170(v9, &v62);
            goto LABEL_39;
          }
          if ( v9 == &v62 )
            goto LABEL_39;
          v30 = v9[1];
          if ( v30 )
          {
            v31 = 32LL * *(_QWORD *)(v30 - 8);
            v32 = v30 + v31;
            if ( v30 != v30 + v31 )
            {
              do
              {
                v55 = v30;
                v59 = v32 - 32;
                sub_127D120((_QWORD *)(v32 - 24));
                v32 = v59;
                v30 = v55;
              }
              while ( v55 != v59 );
            }
            j_j_j___libc_free_0_0(v30 - 8);
          }
          goto LABEL_53;
        }
      }
      if ( v7 != v13 )
      {
        sub_1698680(v9, &v62);
        v14 = v9 + 5;
        if ( v10 == v9 + 4 )
          break;
        goto LABEL_40;
      }
      if ( v9 != &v62 )
      {
        sub_1698460(v9);
LABEL_53:
        if ( v7 == v62 )
          sub_169C6E0(v9, &v62);
        else
          sub_16986C0(v9, &v62);
      }
LABEL_39:
      v14 = v9 + 5;
      if ( v10 == v9 + 4 )
        break;
LABEL_40:
      v9 = v14;
    }
  }
  *(_QWORD *)(a1 + 16) = 0;
  if ( v7 == v64 )
  {
    v36 = v65;
    if ( v65 )
    {
      v37 = 32LL * *(_QWORD *)(v65 - 8);
      v38 = v65 + v37;
      if ( v65 != v65 + v37 )
      {
        do
        {
          v38 -= 32;
          sub_127D120((_QWORD *)(v38 + 8));
        }
        while ( v36 != v38 );
      }
      j_j_j___libc_free_0_0(v36 - 8);
    }
  }
  else
  {
    sub_1698460(&v64);
  }
  if ( v7 == v62 )
  {
    v33 = v63;
    if ( v63 )
    {
      v34 = 32LL * *(_QWORD *)(v63 - 8);
      v35 = v63 + v34;
      if ( v63 != v63 + v34 )
      {
        do
        {
          v35 -= 32;
          sub_127D120((_QWORD *)(v35 + 8));
        }
        while ( v33 != v35 );
      }
      j_j_j___libc_free_0_0(v33 - 8);
    }
  }
  else
  {
    sub_1698460(&v62);
  }
}
