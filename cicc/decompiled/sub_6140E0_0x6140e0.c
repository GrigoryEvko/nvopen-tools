// Function: sub_6140E0
// Address: 0x6140e0
//
char *__fastcall sub_6140E0(int a1, __int64 a2)
{
  char v3; // al
  char v4; // si
  size_t v5; // r9
  int v6; // ecx
  int v7; // r13d
  char *v8; // r12
  int v9; // edx
  char *v10; // rbx
  char v11; // r14
  char v12; // al
  char v13; // al
  char v14; // al
  char *v15; // r15
  const char *v17; // rdi
  int v18; // r12d
  int v19; // ebx
  const char **v20; // r15
  char v21; // cl
  size_t v22; // r13
  int v23; // r14d
  char v24; // al
  char *v25; // r12
  int v26; // ecx
  const char *v27; // rbx
  bool v28; // zf
  int v29; // eax
  __int64 v30; // rdx
  int v31; // eax
  int v32; // ecx
  size_t v33; // r9
  int v34; // r10d
  char *v35; // r14
  __int64 v36; // rdx
  size_t v37; // r13
  char *v38; // r12
  const char *v39; // rsi
  size_t v40; // r9
  int v41; // r13d
  __int64 v42; // r12
  __int64 v43; // r14
  char *v44; // [rsp+8h] [rbp-68h]
  __int64 v45; // [rsp+10h] [rbp-60h]
  char v46; // [rsp+1Ch] [rbp-54h]
  int v47; // [rsp+1Ch] [rbp-54h]
  int v48; // [rsp+20h] [rbp-50h]
  char *v49; // [rsp+20h] [rbp-50h]
  char *v50; // [rsp+28h] [rbp-48h]
  unsigned int v51; // [rsp+28h] [rbp-48h]
  __int64 v52; // [rsp+30h] [rbp-40h]
  unsigned int v53; // [rsp+30h] [rbp-40h]
  unsigned int n; // [rsp+38h] [rbp-38h]
  size_t na; // [rsp+38h] [rbp-38h]

  v3 = 0;
  v4 = 0;
  v5 = 0;
  v6 = 0;
  v7 = a1;
  v8 = 0;
  v9 = dword_4CF8030;
  v10 = s1;
  while ( 1 )
  {
    if ( !v10 )
      goto LABEL_8;
    v11 = *v10;
    if ( *v10 )
    {
      if ( v3 )
        s1 = v10;
      if ( v4 )
        dword_4CF8030 = v9;
      if ( dword_4CF8188 <= 0 )
        goto LABEL_62;
      v50 = v8;
      v17 = v10;
      v18 = dword_4CF8188;
      v48 = v7;
      v19 = v6;
      v20 = (const char **)&unk_4CF81A8;
      v21 = v11;
      v22 = v5;
      v23 = 0;
      do
      {
        if ( v19 )
        {
          if ( *v20 && (const char *)v22 == v20[2] )
          {
            v52 = a2;
            v46 = v21;
            v31 = strncmp(v17, *v20, v22);
            a2 = v52;
            if ( !v31 )
              goto LABEL_36;
            v21 = v46;
          }
        }
        else
        {
          v24 = *((_BYTE *)v20 + 8);
          if ( v24 && v21 == v24 )
          {
LABEL_36:
            v25 = v50;
            v26 = v19;
            v7 = v48;
            v27 = v17;
            v15 = (char *)&unk_4CF81A0 + 40 * v23;
            goto LABEL_37;
          }
        }
        ++v23;
        v20 += 5;
      }
      while ( v23 != v18 );
      v32 = v19;
      v33 = v22;
      v7 = v48;
      v34 = v23;
      v27 = v17;
      if ( !v32 )
        goto LABEL_62;
      v35 = (char *)&unk_4CF81A0;
      v49 = v50;
      v15 = 0;
      v36 = 5LL * (unsigned int)(v34 - 1);
      v47 = v7;
      v37 = v33;
      v44 = (char *)&unk_4CF81A0;
      v53 = 0;
      v38 = (char *)&unk_4CF81C8 + 40 * (unsigned int)(v34 - 1);
      n = 0;
      v51 = v32;
      v45 = a2;
      do
      {
        v39 = (const char *)*((_QWORD *)v35 + 1);
        if ( v39 && !strncmp(v17, v39, v37) )
        {
          v15 = v35;
          if ( n )
            v53 = n;
          else
            n = v51;
        }
        v35 += 40;
      }
      while ( v38 != v35 );
      v40 = v37;
      v25 = v49;
      v26 = v51;
      v7 = v47;
      a2 = v45;
      if ( v53 )
      {
        na = v40;
        v41 = 0;
        v42 = sub_67EA70(923, v17, v36, v51, v45);
        while ( dword_4CF8188 > v41 )
        {
          v43 = *((_QWORD *)v44 + 1);
          if ( v43 && !strncmp(v17, *((const char **)v44 + 1), na) )
            sub_67DC60(v42, 924, v43);
          v44 += 40;
          ++v41;
        }
        sub_685AB0(v42);
      }
      if ( !v15 || (n & 1) == 0 )
        goto LABEL_62;
LABEL_37:
      v28 = v15[19] == 0;
      byte_4CF8060[*(int *)v15] = 1;
      if ( v28 )
        goto LABEL_62;
      if ( !v15[18] )
      {
        if ( !v25 || !*v25 )
        {
          src = 0;
          if ( v26 )
          {
            ++dword_4CF8030;
            s1 = 0;
          }
          else
          {
            s1 = (char *)(v27 + 1);
          }
          return v15;
        }
LABEL_62:
        sub_60F8D0(v7, a2);
      }
      if ( v26 )
      {
        if ( *v25 == 61 )
        {
          v28 = v25[1] == 0;
          src = v25 + 1;
          if ( v28 )
            goto LABEL_62;
          v29 = dword_4CF8030;
LABEL_44:
          s1 = 0;
          dword_4CF8030 = v29 + 1;
          return v15;
        }
        v30 = ++dword_4CF8030;
        v29 = dword_4CF8030;
        if ( dword_4CF8030 >= v7 )
          sub_684920(2666);
      }
      else
      {
        v29 = dword_4CF8030;
        if ( v27[1] )
        {
          src = (char *)(v27 + 1);
          goto LABEL_44;
        }
        v29 = dword_4CF8030 + 1;
        dword_4CF8030 = v29;
        if ( v7 <= v29 )
          goto LABEL_62;
        v30 = v29;
      }
      src = *(char **)(a2 + 8 * v30);
      goto LABEL_44;
    }
    ++v9;
    v4 = 1;
LABEL_8:
    if ( a1 <= v9 )
      break;
    v10 = *(char **)(a2 + 8LL * v9);
    if ( *v10 != 45 )
      goto LABEL_19;
    v12 = v10[1];
    if ( v12 == 45 )
    {
      v13 = v10[2];
      if ( !v13 )
      {
        s1 = *(char **)(a2 + 8LL * v9);
        v15 = 0;
        dword_4CF8030 = v9 + 1;
        return v15;
      }
      v10 += 2;
      v8 = v10;
      if ( v13 == 61 )
      {
        v5 = 0;
      }
      else
      {
        do
          v14 = *++v8;
        while ( v14 != 61 && v14 );
        v5 = v8 - v10;
      }
      v6 = 1;
    }
    else
    {
      if ( !v12 )
        goto LABEL_19;
      ++v10;
    }
    v3 = 1;
  }
  if ( v3 )
LABEL_19:
    s1 = v10;
  if ( v4 )
    dword_4CF8030 = v9;
  return 0;
}
