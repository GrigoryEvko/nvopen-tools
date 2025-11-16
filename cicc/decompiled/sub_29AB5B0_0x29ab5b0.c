// Function: sub_29AB5B0
// Address: 0x29ab5b0
//
void __fastcall sub_29AB5B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  unsigned __int64 v6; // r12
  int v7; // eax
  __int64 v8; // r12
  unsigned int v9; // eax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r15
  unsigned int v13; // r13d
  unsigned int v14; // eax
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r13
  unsigned int v18; // r15d
  unsigned int v19; // eax
  __int64 v20; // r13
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // r9
  int v25; // esi
  __int64 v26; // rdi
  __int64 v27; // rdx
  int v28; // esi
  unsigned int v29; // ecx
  __int64 *v30; // rax
  __int64 v31; // r10
  unsigned __int64 v32; // rdx
  __int64 v33; // r12
  __int64 v34; // r15
  _BYTE *v35; // r13
  __int64 v36; // rax
  unsigned __int64 v37; // rdi
  int v38; // eax
  _DWORD *v39; // rax
  __int64 v40; // r15
  __int64 v41; // rax
  unsigned int v42; // r15d
  unsigned int v43; // r15d
  unsigned int v44; // eax
  __int64 v45; // rdx
  _DWORD *v46; // rax
  _DWORD *v47; // rdx
  __int64 v49; // [rsp+0h] [rbp-120h]
  __int64 v51; // [rsp+38h] [rbp-E8h] BYREF
  _BYTE *v52; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v53; // [rsp+48h] [rbp-D8h]
  _BYTE v54[16]; // [rsp+50h] [rbp-D0h] BYREF
  void *s; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v56; // [rsp+68h] [rbp-B8h]
  _DWORD v57[8]; // [rsp+70h] [rbp-B0h] BYREF
  _BYTE *v58; // [rsp+90h] [rbp-90h] BYREF
  __int64 v59; // [rsp+98h] [rbp-88h]
  _BYTE v60[64]; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v61; // [rsp+E0h] [rbp-40h]
  char v62; // [rsp+E8h] [rbp-38h]

  v4 = a2;
  v6 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v6 == a2 + 48 )
  {
    v8 = 0;
  }
  else
  {
    if ( !v6 )
      BUG();
    v7 = *(unsigned __int8 *)(v6 - 24);
    v8 = v6 - 24;
    if ( (unsigned int)(v7 - 30) >= 0xB )
      v8 = 0;
  }
  v9 = sub_B46E30(v8);
  v12 = v9;
  v13 = v9;
  s = v57;
  v56 = 0x800000000LL;
  if ( v9 > 8 )
  {
    sub_C8D5F0((__int64)&s, v57, v9, 4u, v10, v11);
    memset(s, 0, 4 * v12);
    LODWORD(v56) = v12;
  }
  else
  {
    if ( v9 )
    {
      v40 = 4LL * v9;
      if ( v40 )
      {
        if ( (unsigned int)v40 < 8 )
        {
          if ( ((4 * (_BYTE)v9) & 4) != 0 )
          {
            v57[0] = 0;
            *(_DWORD *)((char *)&v57[-1] + (unsigned int)v40) = 0;
          }
          else if ( (_DWORD)v40 )
          {
            LOBYTE(v57[0]) = 0;
          }
        }
        else
        {
          v41 = (unsigned int)v40;
          v42 = v40 - 1;
          *(_QWORD *)((char *)&v57[-2] + v41) = 0;
          if ( v42 >= 8 )
          {
            v43 = v42 & 0xFFFFFFF8;
            v44 = 0;
            do
            {
              v45 = v44;
              v44 += 8;
              *(_QWORD *)((char *)v57 + v45) = 0;
            }
            while ( v44 < v43 );
          }
        }
      }
    }
    LODWORD(v56) = v13;
  }
  v62 = 0;
  v59 = 0x400000000LL;
  v58 = v60;
  v61 = 0;
  v14 = sub_B46E30(v8);
  v17 = v14;
  v53 = 0x400000000LL;
  v18 = v14;
  v52 = v54;
  if ( v14 <= 4 )
  {
    if ( v14 )
    {
      v39 = v54;
      do
      {
        *v39++ = -1;
        --v17;
      }
      while ( v17 );
    }
    LODWORD(v53) = v18;
    v19 = sub_B46E30(v8);
    if ( v19 )
    {
LABEL_11:
      v20 = v19;
      v21 = 0;
      while ( 1 )
      {
        LODWORD(v51) = v21;
        v22 = sub_B46EC0(v8, v21);
        v25 = *(_DWORD *)(a3 + 24);
        v26 = *(_QWORD *)(a3 + 8);
        v27 = v22;
        if ( !v25 )
          goto LABEL_12;
        v28 = v25 - 1;
        v29 = v28 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v30 = (__int64 *)(v26 + 16LL * v29);
        v31 = *v30;
        if ( v27 != *v30 )
        {
          v38 = 1;
          while ( v31 != -4096 )
          {
            v23 = (unsigned int)(v38 + 1);
            v29 = v28 & (v38 + v29);
            v30 = (__int64 *)(v26 + 16LL * v29);
            v31 = *v30;
            if ( v27 == *v30 )
              goto LABEL_15;
            v38 = v23;
          }
          goto LABEL_12;
        }
LABEL_15:
        v32 = v30[1];
        if ( v32 )
        {
          ++v21;
          sub_FE8630((__int64)&v58, (unsigned int *)&v51, v32, 1u, v23, v24);
          if ( v20 == v21 )
          {
LABEL_17:
            v4 = a2;
            goto LABEL_18;
          }
        }
        else
        {
LABEL_12:
          *(_DWORD *)&v52[4 * v21++] = 0;
          if ( v20 == v21 )
            goto LABEL_17;
        }
      }
    }
    goto LABEL_43;
  }
  sub_C8D5F0((__int64)&v52, v54, v14, 4u, v15, v16);
  v46 = v52;
  v47 = &v52[4 * v17];
  do
  {
    if ( v46 )
      *v46 = -1;
    ++v46;
  }
  while ( v47 != v46 );
  LODWORD(v53) = v17;
  v19 = sub_B46E30(v8);
  if ( v19 )
    goto LABEL_11;
LABEL_18:
  if ( !v61 )
  {
LABEL_43:
    sub_FF6650(a4, v4, (__int64)&v52);
    v37 = (unsigned __int64)v52;
    if ( v52 == v54 )
      goto LABEL_25;
LABEL_24:
    _libc_free(v37);
    goto LABEL_25;
  }
  sub_FE9FC0((__int64)&v58);
  if ( (_DWORD)v59 )
  {
    v49 = v8;
    v33 = 0;
    v34 = 16LL * (unsigned int)v59;
    do
    {
      v35 = &v58[v33];
      v33 += 16;
      *((_DWORD *)s + *((unsigned int *)v35 + 1)) = *((_QWORD *)v35 + 1);
      sub_F02DB0(&v51, *((_DWORD *)v35 + 2), v61);
      *(_DWORD *)&v52[4 * *((unsigned int *)v35 + 1)] = v51;
    }
    while ( v33 != v34 );
    v8 = v49;
  }
  sub_FF6650(a4, v4, (__int64)&v52);
  v51 = sub_BD5C60(v8);
  v36 = sub_B8C150(&v51, (unsigned int *)s, (unsigned int)v56, 0);
  sub_B99FD0(v8, 2u, v36);
  v37 = (unsigned __int64)v52;
  if ( v52 != v54 )
    goto LABEL_24;
LABEL_25:
  if ( v58 != v60 )
    _libc_free((unsigned __int64)v58);
  if ( s != v57 )
    _libc_free((unsigned __int64)s);
}
