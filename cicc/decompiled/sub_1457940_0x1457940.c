// Function: sub_1457940
// Address: 0x1457940
//
__int64 __fastcall sub_1457940(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v5; // rbx
  unsigned int v6; // eax
  __int64 *v7; // r9
  int *v8; // r8
  unsigned int v9; // r12d
  __int64 v10; // rax
  unsigned int v11; // ecx
  unsigned __int64 v12; // r15
  __int64 v13; // rax
  unsigned int v14; // r13d
  unsigned int v16; // eax
  __int64 *v17; // r9
  int *v18; // r8
  __int64 v19; // rax
  unsigned int v20; // ecx
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned int v23; // edx
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // edx
  __int64 v27; // rax
  int v28; // eax
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rax
  __int64 *v33; // [rsp+8h] [rbp-78h]
  int *v34; // [rsp+10h] [rbp-70h]
  unsigned int v35; // [rsp+18h] [rbp-68h]
  int v36; // [rsp+2Ch] [rbp-54h] BYREF
  __int64 v37; // [rsp+30h] [rbp-50h] BYREF
  __int64 v38; // [rsp+38h] [rbp-48h] BYREF
  unsigned __int64 v39; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v40; // [rsp+48h] [rbp-38h]

  v4 = a4;
  v5 = a3;
  v40 = 1;
  v39 = 0;
  if ( a2 == 40 )
  {
    v4 = a3;
    v5 = a4;
    goto LABEL_26;
  }
  if ( a2 > 0x28 )
  {
    v9 = 0;
    if ( a2 != 41 )
      return v9;
    v4 = a3;
    v5 = a4;
    goto LABEL_5;
  }
  if ( a2 == 38 )
  {
LABEL_26:
    v16 = sub_1457900(a1, v5, &v38, &v37, &v36);
    v17 = &v37;
    v18 = &v36;
    v9 = v16;
    if ( (_BYTE)v16 && !*(_WORD *)(v38 + 24) && v4 == v37 )
    {
      v19 = *(_QWORD *)(v38 + 32);
      if ( v40 <= 0x40 && (v20 = *(_DWORD *)(v19 + 32), v20 <= 0x40) )
      {
        v30 = *(_QWORD *)(v19 + 24);
        v40 = v20;
        v39 = v30;
        sub_1453710(&v39);
      }
      else
      {
        sub_16A51C0(&v39, v19 + 24);
        v17 = &v37;
        v18 = &v36;
      }
      if ( (v36 & 4) != 0 )
      {
        v12 = v39;
        v21 = 1LL << ((unsigned __int8)v40 - 1);
        if ( v40 > 0x40 )
        {
          if ( (*(_QWORD *)(v39 + 8LL * ((v40 - 1) >> 6)) & v21) == 0 )
          {
            v33 = v17;
            v34 = v18;
            v35 = v40;
            v28 = sub_16A57B0(&v39);
            v18 = v34;
            v17 = v33;
            if ( v35 != v28 )
              goto LABEL_20;
          }
        }
        else if ( (v21 & v39) == 0 && v39 )
        {
          return v9;
        }
      }
    }
    v9 = sub_1457900(a1, v4, &v38, v17, v18);
    if ( (_BYTE)v9 )
    {
      v14 = v40;
      if ( !*(_WORD *)(v38 + 24) && v5 == v37 )
      {
        v22 = *(_QWORD *)(v38 + 32);
        if ( v40 <= 0x40 && (v23 = *(_DWORD *)(v22 + 32), v23 <= 0x40) )
        {
          v32 = *(_QWORD *)(v22 + 24);
          v40 = v23;
          v39 = v32;
          sub_1453710(&v39);
        }
        else
        {
          sub_16A51C0(&v39, v22 + 24);
        }
        v14 = v40;
        if ( (v36 & 4) != 0 )
        {
          v24 = v39;
          if ( v40 > 0x40 )
            v24 = *(_QWORD *)(v39 + 8LL * ((v40 - 1) >> 6));
          LOBYTE(v9) = (v24 & (1LL << ((unsigned __int8)v40 - 1))) != 0;
          goto LABEL_18;
        }
      }
      goto LABEL_17;
    }
LABEL_46:
    v14 = v40;
    goto LABEL_18;
  }
  if ( a2 != 39 )
    return 0;
LABEL_5:
  v6 = sub_1457900(a1, v5, &v38, &v37, &v36);
  v7 = &v37;
  v8 = &v36;
  v9 = v6;
  if ( !(_BYTE)v6 || *(_WORD *)(v38 + 24) || v4 != v37 )
    goto LABEL_14;
  v10 = *(_QWORD *)(v38 + 32);
  if ( v40 <= 0x40 && (v11 = *(_DWORD *)(v10 + 32), v11 <= 0x40) )
  {
    v29 = *(_QWORD *)(v10 + 24);
    v40 = v11;
    v39 = v29;
    sub_1453710(&v39);
  }
  else
  {
    sub_16A51C0(&v39, v10 + 24);
    v7 = &v37;
    v8 = &v36;
  }
  if ( (v36 & 4) == 0 )
    goto LABEL_14;
  v12 = v39;
  v13 = 1LL << ((unsigned __int8)v40 - 1);
  if ( v40 <= 0x40 )
  {
    if ( (v13 & v39) == 0 )
      return 1;
LABEL_14:
    v9 = sub_1457900(a1, v4, &v38, v7, v8);
    if ( (_BYTE)v9 )
    {
      v14 = v40;
      if ( !*(_WORD *)(v38 + 24) && v5 == v37 )
      {
        v25 = *(_QWORD *)(v38 + 32);
        if ( v40 <= 0x40 && (v26 = *(_DWORD *)(v25 + 32), v26 <= 0x40) )
        {
          v31 = *(_QWORD *)(v25 + 24);
          v40 = v26;
          v39 = v31;
          sub_1453710(&v39);
        }
        else
        {
          sub_16A51C0(&v39, v25 + 24);
        }
        v14 = v40;
        if ( (v36 & 4) != 0 )
        {
          v12 = v39;
          v27 = 1LL << ((unsigned __int8)v40 - 1);
          if ( v40 <= 0x40 )
          {
            if ( (v27 & v39) == 0 )
            {
              LOBYTE(v9) = v39 == 0;
              return v9;
            }
            return 1;
          }
          if ( (*(_QWORD *)(v39 + 8LL * ((v40 - 1) >> 6)) & v27) == 0 )
            LOBYTE(v9) = v14 == (unsigned int)sub_16A57B0(&v39);
          goto LABEL_20;
        }
      }
LABEL_17:
      v9 = 0;
LABEL_18:
      if ( v14 <= 0x40 )
        return v9;
      v12 = v39;
      goto LABEL_20;
    }
    goto LABEL_46;
  }
  if ( (*(_QWORD *)(v39 + 8LL * ((v40 - 1) >> 6)) & v13) != 0 )
    goto LABEL_14;
LABEL_20:
  if ( v12 )
    j_j___libc_free_0_0(v12);
  return v9;
}
