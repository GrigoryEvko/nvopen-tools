// Function: sub_2B1E780
// Address: 0x2b1e780
//
__int64 __fastcall sub_2B1E780(_QWORD *a1, _BYTE *a2, __int64 a3, __int64 a4, _BYTE *a5)
{
  __int64 v5; // rdx
  _QWORD *v6; // rax
  __int64 v7; // rdx
  _QWORD *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  _QWORD *v11; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  _QWORD *v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdx
  _QWORD *v19; // r9
  int v20; // edx
  unsigned int v21; // eax
  _QWORD *v22; // rdi
  _BYTE *v23; // r8
  __int64 v24; // rcx
  __int64 v25; // rdx
  _QWORD *v26; // r9
  int v27; // edx
  unsigned int v28; // eax
  _QWORD *v29; // rdi
  int v30; // edi
  int v31; // r10d
  int v32; // edi
  int v33; // r10d

  if ( (unsigned __int8)(*a2 - 82) > 1u )
  {
    LOBYTE(a5) = *a2 == 94 || *a2 == 91;
    if ( !(_BYTE)a5 )
      return (unsigned int)a5;
    v13 = a1[1];
    if ( *(_DWORD *)(v13 + 16) )
    {
      v24 = *(_QWORD *)(v13 + 8);
      v25 = *(unsigned int *)(v13 + 24);
      v26 = (_QWORD *)(v24 + 8 * v25);
      if ( (_DWORD)v25 )
      {
        v27 = v25 - 1;
        v28 = v27 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v29 = (_QWORD *)(v24 + 8LL * v28);
        a5 = (_BYTE *)*v29;
        if ( (_BYTE *)*v29 == a2 )
        {
LABEL_27:
          LOBYTE(a5) = v29 != v26;
          return (unsigned int)a5;
        }
        v32 = 1;
        while ( a5 != (_BYTE *)-4096LL )
        {
          v33 = v32 + 1;
          v28 = v27 & (v32 + v28);
          v29 = (_QWORD *)(v24 + 8LL * v28);
          a5 = (_BYTE *)*v29;
          if ( (_BYTE *)*v29 == a2 )
            goto LABEL_27;
          v32 = v33;
        }
      }
      return 0;
    }
    v6 = *(_QWORD **)(v13 + 32);
    v14 = 8LL * *(unsigned int *)(v13 + 40);
    v8 = &v6[(unsigned __int64)v14 / 8];
    v9 = v14 >> 3;
    v15 = v14 >> 5;
    if ( v15 )
    {
      v16 = &v6[4 * v15];
      while ( (_BYTE *)*v6 != a2 )
      {
        if ( (_BYTE *)v6[1] == a2 )
        {
LABEL_33:
          ++v6;
          goto LABEL_10;
        }
        if ( (_BYTE *)v6[2] == a2 )
        {
LABEL_34:
          LOBYTE(a5) = v8 != v6 + 2;
          return (unsigned int)a5;
        }
        if ( (_BYTE *)v6[3] == a2 )
        {
LABEL_35:
          LOBYTE(a5) = v8 != v6 + 3;
          return (unsigned int)a5;
        }
        v6 += 4;
        if ( v6 == v16 )
        {
          v9 = v8 - v6;
          goto LABEL_43;
        }
      }
    }
    else
    {
LABEL_43:
      if ( v9 == 2 )
        goto LABEL_38;
      if ( v9 != 3 )
      {
LABEL_31:
        if ( v9 != 1 )
          return 0;
LABEL_40:
        LODWORD(a5) = 0;
        if ( (_BYTE *)*v6 != a2 )
          return (unsigned int)a5;
        goto LABEL_10;
      }
      if ( (_BYTE *)*v6 != a2 )
      {
LABEL_37:
        ++v6;
        goto LABEL_38;
      }
    }
    LOBYTE(a5) = v6 != v8;
    return (unsigned int)a5;
  }
  v5 = *a1;
  if ( !*(_DWORD *)(*a1 + 16LL) )
  {
    v6 = *(_QWORD **)(v5 + 32);
    v7 = 8LL * *(unsigned int *)(v5 + 40);
    v8 = &v6[(unsigned __int64)v7 / 8];
    v9 = v7 >> 3;
    v10 = v7 >> 5;
    if ( v10 )
    {
      v11 = &v6[4 * v10];
      while ( (_BYTE *)*v6 != a2 )
      {
        if ( (_BYTE *)v6[1] == a2 )
          goto LABEL_33;
        if ( (_BYTE *)v6[2] == a2 )
          goto LABEL_34;
        if ( (_BYTE *)v6[3] == a2 )
          goto LABEL_35;
        v6 += 4;
        if ( v11 == v6 )
        {
          v9 = v8 - v6;
          goto LABEL_29;
        }
      }
      goto LABEL_10;
    }
LABEL_29:
    if ( v9 != 2 )
    {
      if ( v9 != 3 )
        goto LABEL_31;
      if ( (_BYTE *)*v6 == a2 )
      {
LABEL_10:
        LOBYTE(a5) = v8 != v6;
        return (unsigned int)a5;
      }
      goto LABEL_37;
    }
LABEL_38:
    if ( (_BYTE *)*v6 != a2 )
    {
      ++v6;
      goto LABEL_40;
    }
    goto LABEL_10;
  }
  v17 = *(_QWORD *)(v5 + 8);
  v18 = *(unsigned int *)(v5 + 24);
  v19 = (_QWORD *)(v17 + 8 * v18);
  if ( (_DWORD)v18 )
  {
    v20 = v18 - 1;
    v21 = v20 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v22 = (_QWORD *)(v17 + 8LL * v21);
    v23 = (_BYTE *)*v22;
    if ( a2 == (_BYTE *)*v22 )
    {
LABEL_24:
      LOBYTE(v23) = v19 != v22;
      return (unsigned int)v23;
    }
    v30 = 1;
    while ( v23 != (_BYTE *)-4096LL )
    {
      v31 = v30 + 1;
      v21 = v20 & (v30 + v21);
      v22 = (_QWORD *)(v17 + 8LL * v21);
      v23 = (_BYTE *)*v22;
      if ( (_BYTE *)*v22 == a2 )
        goto LABEL_24;
      v30 = v31;
    }
  }
  return 0;
}
