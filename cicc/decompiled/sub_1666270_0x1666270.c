// Function: sub_1666270
// Address: 0x1666270
//
void __fastcall sub_1666270(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rcx
  char v6; // dl
  __int64 v7; // r12
  _BYTE *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdi
  unsigned __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rax
  _QWORD *v14; // rbx
  __int64 v15; // rax
  unsigned __int64 v16; // r15
  char *v17; // rax
  int v18; // r9d
  char *v19; // r8
  __int64 v20; // r9
  char *v21; // rdi
  __int64 v22; // rdx
  char *v23; // rax
  __int64 v24; // rcx
  char v25; // dl
  __int64 v26; // rcx
  char v27; // dl
  __int64 v28; // rcx
  char v29; // dl
  __int64 v30; // rcx
  char v31; // dl
  __int64 v32; // r15
  __int64 v33; // rdx
  char v34; // al
  char v35; // cl
  __int64 v36; // r8
  _BYTE *v37; // rax
  const char *v38; // rax
  unsigned __int64 v39; // rdi
  __int64 v40; // rcx
  char v41; // dl
  __int64 v42; // rdx
  int v43; // edi
  __int64 v44; // rax
  char *v45; // rax
  char *v46; // rsi
  __int64 v47; // rcx
  char v48; // dl
  __int64 v49; // rcx
  char v50; // dl
  __int64 v51; // rcx
  char v52; // dl
  const char *v53; // rax
  __int64 v54; // [rsp+8h] [rbp-E8h] BYREF
  _QWORD v55[2]; // [rsp+10h] [rbp-E0h] BYREF
  char v56; // [rsp+20h] [rbp-D0h]
  char v57; // [rsp+21h] [rbp-CFh]
  char *v58; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v59; // [rsp+38h] [rbp-B8h]
  char v60; // [rsp+40h] [rbp-B0h] BYREF
  char v61; // [rsp+41h] [rbp-AFh]

  v4 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v5 = **(_QWORD **)(a2 - 24 * v4);
  v6 = *(_BYTE *)(v5 + 8);
  if ( v6 == 16 )
    v6 = *(_BYTE *)(**(_QWORD **)(v5 + 16) + 8LL);
  if ( v6 == 15 )
  {
    v10 = *(_QWORD *)(a2 + 56);
    v11 = *(unsigned __int8 *)(v10 + 8);
    if ( (unsigned __int8)v11 > 0xFu || (v12 = 35454, !_bittest64(&v12, v11)) )
    {
      if ( (unsigned int)(v11 - 13) > 1 && (_DWORD)v11 != 16 || !sub_16435F0(v10, 0) )
      {
        v61 = 1;
        v58 = "GEP into unsized type!";
        v60 = 3;
        sub_164FF40(a1, (__int64)&v58);
        if ( !*a1 )
          return;
LABEL_49:
        sub_164FA80(a1, a2);
        return;
      }
      v4 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    }
    v58 = &v60;
    v59 = 0x1000000000LL;
    v13 = 24 * (1 - v4);
    v14 = (_QWORD *)(a2 + v13);
    v15 = -v13;
    v16 = 0xAAAAAAAAAAAAAAABLL * (v15 >> 3);
    if ( (unsigned __int64)v15 > 0x180 )
    {
      sub_16CD150(&v58, &v60, 0xAAAAAAAAAAAAAAABLL * (v15 >> 3), 8);
      v19 = v58;
      v18 = v59;
      v17 = &v58[8 * (unsigned int)v59];
    }
    else
    {
      v17 = &v60;
      v18 = 0;
      v19 = &v60;
    }
    if ( (_QWORD *)a2 != v14 )
    {
      do
      {
        if ( v17 )
          *(_QWORD *)v17 = *v14;
        v14 += 3;
        v17 += 8;
      }
      while ( (_QWORD *)a2 != v14 );
      v19 = v58;
      v18 = v59;
    }
    LODWORD(v59) = v16 + v18;
    v20 = (unsigned int)(v16 + v18);
    v21 = &v19[8 * v20];
    v22 = (8 * v20) >> 3;
    if ( (8 * v20) >> 5 )
    {
      v23 = v19;
      while ( 1 )
      {
        v30 = **(_QWORD **)v23;
        v31 = *(_BYTE *)(v30 + 8);
        if ( v31 == 16 )
          v31 = *(_BYTE *)(**(_QWORD **)(v30 + 16) + 8LL);
        if ( v31 != 11 )
          goto LABEL_33;
        v24 = **((_QWORD **)v23 + 1);
        v25 = *(_BYTE *)(v24 + 8);
        if ( v25 == 16 )
          v25 = *(_BYTE *)(**(_QWORD **)(v24 + 16) + 8LL);
        if ( v25 != 11 )
        {
          v23 += 8;
          goto LABEL_33;
        }
        v26 = **((_QWORD **)v23 + 2);
        v27 = *(_BYTE *)(v26 + 8);
        if ( v27 == 16 )
          v27 = *(_BYTE *)(**(_QWORD **)(v26 + 16) + 8LL);
        if ( v27 != 11 )
        {
          v23 += 16;
          goto LABEL_33;
        }
        v28 = **((_QWORD **)v23 + 3);
        v29 = *(_BYTE *)(v28 + 8);
        if ( v29 == 16 )
          v29 = *(_BYTE *)(**(_QWORD **)(v28 + 16) + 8LL);
        if ( v29 != 11 )
          break;
        v23 += 32;
        if ( v23 == &v19[32 * ((8 * v20) >> 5)] )
        {
          v22 = (v21 - v23) >> 3;
          goto LABEL_61;
        }
      }
      v23 += 24;
LABEL_33:
      if ( v21 != v23 )
      {
        v57 = 1;
        v38 = "GEP indexes must be integers";
        goto LABEL_51;
      }
LABEL_34:
      v32 = sub_15F9F50(*(_QWORD *)(a2 + 56), (__int64)v19, v20);
      if ( v32 )
      {
        v33 = *(_QWORD *)a2;
        v34 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
        v35 = v34;
        if ( v34 == 16 )
          v35 = *(_BYTE *)(**(_QWORD **)(v33 + 16) + 8LL);
        if ( v35 != 15 || v32 != *(_QWORD *)(a2 + 64) )
        {
          v57 = 1;
          v55[0] = "GEP is not of right type for indices!";
          v56 = 3;
          sub_164FF40(a1, (__int64)v55);
          if ( *a1 )
          {
            sub_164FA80(a1, a2);
            v36 = *a1;
            v37 = *(_BYTE **)(*a1 + 24);
            if ( (unsigned __int64)v37 >= *(_QWORD *)(*a1 + 16) )
            {
              v36 = sub_16E7DE0(*a1, 32);
            }
            else
            {
              *(_QWORD *)(v36 + 24) = v37 + 1;
              *v37 = 32;
            }
            sub_154E060(v32, v36, 0, 0);
          }
          goto LABEL_53;
        }
        if ( v34 != 16 )
          goto LABEL_77;
        v42 = *(_QWORD *)(v33 + 32);
        v43 = v42;
        v44 = **(_QWORD **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        if ( *(_BYTE *)(v44 + 8) == 16 && *(_DWORD *)(v44 + 32) != (_DWORD)v42 )
        {
          v54 = a2;
          v53 = "Vector GEP result width doesn't match operand's";
          v57 = 1;
          goto LABEL_92;
        }
        v45 = v58;
        v46 = &v58[8 * (unsigned int)v59];
        if ( v58 == v46 )
        {
LABEL_77:
          sub_1663F80((__int64)a1, a2);
          v39 = (unsigned __int64)v58;
          if ( v58 != &v60 )
            goto LABEL_54;
          return;
        }
        while ( 1 )
        {
          v47 = **(_QWORD **)v45;
          v48 = *(_BYTE *)(v47 + 8);
          if ( v48 == 16 )
          {
            if ( v43 != *(_DWORD *)(v47 + 32) )
            {
              v54 = a2;
              v53 = "Invalid GEP index vector width";
              v57 = 1;
LABEL_92:
              v55[0] = v53;
              v56 = 3;
              sub_165B6D0(a1, (__int64)v55, &v54);
LABEL_53:
              v39 = (unsigned __int64)v58;
              if ( v58 != &v60 )
LABEL_54:
                _libc_free(v39);
              return;
            }
            v48 = *(_BYTE *)(**(_QWORD **)(v47 + 16) + 8LL);
          }
          if ( v48 != 11 )
          {
            v57 = 1;
            v55[0] = "All GEP indices should be of integer type";
            v56 = 3;
            sub_164FF40(a1, (__int64)v55);
            goto LABEL_53;
          }
          v45 += 8;
          if ( v46 == v45 )
            goto LABEL_77;
        }
      }
      v57 = 1;
      v38 = "Invalid indices for GEP pointer type!";
LABEL_51:
      v55[0] = v38;
      v56 = 3;
      sub_164FF40(a1, (__int64)v55);
      if ( *a1 )
        sub_164FA80(a1, a2);
      goto LABEL_53;
    }
    v23 = v19;
LABEL_61:
    if ( v22 != 2 )
    {
      if ( v22 != 3 )
      {
        if ( v22 != 1 )
          goto LABEL_34;
        goto LABEL_64;
      }
      v49 = **(_QWORD **)v23;
      v50 = *(_BYTE *)(v49 + 8);
      if ( v50 == 16 )
        v50 = *(_BYTE *)(**(_QWORD **)(v49 + 16) + 8LL);
      if ( v50 != 11 )
        goto LABEL_33;
      v23 += 8;
    }
    v51 = **(_QWORD **)v23;
    v52 = *(_BYTE *)(v51 + 8);
    if ( v52 == 16 )
      v52 = *(_BYTE *)(**(_QWORD **)(v51 + 16) + 8LL);
    if ( v52 != 11 )
      goto LABEL_33;
    v23 += 8;
LABEL_64:
    v40 = **(_QWORD **)v23;
    v41 = *(_BYTE *)(v40 + 8);
    if ( v41 == 16 )
      v41 = *(_BYTE *)(**(_QWORD **)(v40 + 16) + 8LL);
    if ( v41 == 11 )
      goto LABEL_34;
    goto LABEL_33;
  }
  v7 = *a1;
  v61 = 1;
  v58 = "GEP base pointer is not a vector or a vector of pointers";
  v60 = 3;
  if ( !v7 )
  {
    *((_BYTE *)a1 + 72) = 1;
    return;
  }
  sub_16E2CE0(&v58, v7);
  v8 = *(_BYTE **)(v7 + 24);
  if ( (unsigned __int64)v8 >= *(_QWORD *)(v7 + 16) )
  {
    sub_16E7DE0(v7, 10);
  }
  else
  {
    *(_QWORD *)(v7 + 24) = v8 + 1;
    *v8 = 10;
  }
  v9 = *a1;
  *((_BYTE *)a1 + 72) = 1;
  if ( v9 )
    goto LABEL_49;
}
