// Function: sub_F90360
// Address: 0xf90360
//
char __fastcall sub_F90360(__int64 a1, _BYTE *a2, __int64 a3)
{
  _BYTE *v4; // rbx
  __int64 *v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned __int64 v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdi
  _QWORD *v12; // r14
  _QWORD *v13; // rdx
  _QWORD *v14; // r8
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 *v17; // rax
  __int64 *v18; // rbx
  unsigned int v19; // eax
  __int64 v20; // r12
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdi
  _QWORD *v25; // r15
  _QWORD *v26; // rdx
  _QWORD *v27; // r14
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  __int64 *v30; // rax
  __int64 v31; // r14
  __int64 v32; // rdi
  __int64 v33; // rdx
  __int64 *v34; // rdi
  __int64 v35; // rdx
  __int64 *v36; // rbx
  __int64 v37; // rsi
  __int64 *v38; // r14
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // r12
  __int64 *v43; // r12
  __int64 v44; // rax
  __int64 *v45; // r15
  _QWORD *v46; // rsi
  char v47; // al
  _QWORD *v49; // [rsp+8h] [rbp-88h]
  __int64 *v51; // [rsp+20h] [rbp-70h] BYREF
  __int64 v52; // [rsp+28h] [rbp-68h]
  _BYTE v53[96]; // [rsp+30h] [rbp-60h] BYREF

  v4 = a2;
  LOBYTE(v5) = sub_B44020((__int64)a2);
  if ( !(_BYTE)v5 )
    return (char)v5;
  v8 = *(unsigned int *)(a3 + 8) + 1LL;
  v51 = (__int64 *)v53;
  v52 = 0x300000000LL;
  if ( v8 <= 3 )
  {
    v9 = *((_QWORD *)a2 + 8);
    if ( v9 )
      goto LABEL_4;
LABEL_21:
    v12 = &qword_4F81430[1];
    v14 = &qword_4F81430[1];
    goto LABEL_6;
  }
  a2 = v53;
  sub_C8D5F0((__int64)&v51, v53, v8, 0x10u, v6, v7);
  v9 = *((_QWORD *)v4 + 8);
  if ( !v9 )
    goto LABEL_21;
LABEL_4:
  v10 = sub_B14240(v9);
  v11 = *((_QWORD *)v4 + 8);
  v12 = (_QWORD *)v10;
  if ( v11 )
  {
    sub_B14240(v11);
    v14 = v13;
  }
  else
  {
    v14 = &qword_4F81430[1];
  }
LABEL_6:
  v15 = (unsigned int)v52;
  v16 = (unsigned int)v52 + 1LL;
  if ( v16 > HIDWORD(v52) )
  {
    a2 = v53;
    v49 = v14;
    sub_C8D5F0((__int64)&v51, v53, v16, 0x10u, (__int64)v14, v7);
    v15 = (unsigned int)v52;
    v14 = v49;
  }
  v17 = &v51[2 * v15];
  *v17 = (__int64)v12;
  v17[1] = (__int64)v14;
  v18 = *(__int64 **)a3;
  v19 = v52 + 1;
  v20 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
  LODWORD(v52) = v52 + 1;
  if ( v18 == (__int64 *)v20 )
  {
LABEL_24:
    v33 = v19;
    v34 = v51;
    while ( 1 )
    {
      v35 = 16 * v33;
      v36 = &v34[(unsigned __int64)v35 / 8];
      v37 = v35 >> 4;
      if ( v35 >> 6 )
      {
        v5 = v34;
        while ( 1 )
        {
          v37 = v5[1];
          if ( *v5 == v37 )
            goto LABEL_32;
          v37 = v5[3];
          if ( v5[2] == v37 )
          {
            v5 += 2;
            goto LABEL_32;
          }
          v37 = v5[5];
          if ( v5[4] == v37 )
          {
            v5 += 4;
            goto LABEL_32;
          }
          v37 = v5[7];
          if ( v5[6] == v37 )
          {
            v5 += 6;
            goto LABEL_32;
          }
          v5 += 8;
          if ( &v34[8 * (v35 >> 6)] == v5 )
          {
            v37 = ((char *)v36 - (char *)v5) >> 4;
            goto LABEL_51;
          }
        }
      }
      v5 = v34;
LABEL_51:
      if ( v37 == 2 )
        goto LABEL_68;
      if ( v37 != 3 )
      {
        if ( v37 != 1 )
          goto LABEL_33;
LABEL_54:
        if ( *v5 != v5[1] )
          goto LABEL_33;
        goto LABEL_32;
      }
      if ( *v5 != v5[1] )
        break;
LABEL_32:
      if ( v36 != v5 )
      {
        if ( v34 != (__int64 *)v53 )
          LOBYTE(v5) = _libc_free(v34, v37);
        return (char)v5;
      }
LABEL_33:
      v38 = v34 + 2;
      v39 = v35 - 16;
      v40 = (v35 - 16) >> 6;
      v41 = v39 >> 4;
      if ( v40 > 0 )
      {
        v42 = (__int64)&v34[8 * v40 + 2];
        while ( (unsigned __int8)sub_B12420(*v34, *v38) )
        {
          if ( !(unsigned __int8)sub_B12420(*v51, v38[2]) )
          {
            v34 = v51;
            v38 += 2;
            goto LABEL_41;
          }
          if ( !(unsigned __int8)sub_B12420(*v51, v38[4]) )
          {
            v34 = v51;
            v38 += 4;
            goto LABEL_41;
          }
          if ( !(unsigned __int8)sub_B12420(*v51, v38[6]) )
          {
            v34 = v51;
            v38 += 6;
            goto LABEL_41;
          }
          v38 += 8;
          v34 = v51;
          if ( (__int64 *)v42 == v38 )
          {
            v41 = ((char *)v36 - (char *)v38) >> 4;
            goto LABEL_57;
          }
        }
        goto LABEL_40;
      }
LABEL_57:
      if ( v41 == 2 )
        goto LABEL_72;
      if ( v41 != 3 )
      {
        if ( v41 == 1 )
        {
LABEL_74:
          v47 = sub_B12420(*v34, *v38);
          v34 = v51;
          if ( !v47 )
            goto LABEL_41;
        }
        v38 = v36;
        goto LABEL_41;
      }
      if ( (unsigned __int8)sub_B12420(*v34, *v38) )
      {
        v34 = v51;
        v38 += 2;
LABEL_72:
        if ( (unsigned __int8)sub_B12420(*v34, *v38) )
        {
          v34 = v51;
          v38 += 2;
          goto LABEL_74;
        }
      }
LABEL_40:
      v34 = v51;
LABEL_41:
      v33 = (unsigned int)v52;
      v43 = v34;
      v44 = 2LL * (unsigned int)v52;
      v45 = &v34[v44];
      if ( &v34[v44] != v34 )
      {
        do
        {
          while ( 1 )
          {
            v46 = (_QWORD *)*v43;
            *v43 = *(_QWORD *)(*v43 + 8);
            if ( v38 == v36 )
              break;
            v43 += 2;
            if ( v45 == v43 )
              goto LABEL_46;
          }
          v43 += 2;
          sub_B14260(v46);
          sub_AA8770(*(_QWORD *)(a1 + 40), (__int64)v46, a1 + 24, 0);
        }
        while ( v45 != v43 );
LABEL_46:
        v34 = v51;
        v33 = (unsigned int)v52;
      }
    }
    v5 += 2;
LABEL_68:
    if ( *v5 != v5[1] )
    {
      v5 += 2;
      goto LABEL_54;
    }
    goto LABEL_32;
  }
  while ( 1 )
  {
    v31 = *v18;
    LOBYTE(v5) = sub_B44020(*v18);
    if ( !(_BYTE)v5 )
      break;
    v32 = *(_QWORD *)(v31 + 64);
    if ( v32 )
    {
      v21 = sub_B14240(v32);
      v24 = *(_QWORD *)(v31 + 64);
      v25 = (_QWORD *)v21;
      if ( v24 )
      {
        sub_B14240(v24);
        v27 = v26;
      }
      else
      {
        v27 = &qword_4F81430[1];
      }
      v28 = (unsigned int)v52;
      v29 = (unsigned int)v52 + 1LL;
      if ( v29 <= HIDWORD(v52) )
        goto LABEL_13;
LABEL_17:
      a2 = v53;
      sub_C8D5F0((__int64)&v51, v53, v29, 0x10u, v22, v23);
      v28 = (unsigned int)v52;
      goto LABEL_13;
    }
    v25 = &qword_4F81430[1];
    v28 = (unsigned int)v52;
    v27 = &qword_4F81430[1];
    v29 = (unsigned int)v52 + 1LL;
    if ( v29 > HIDWORD(v52) )
      goto LABEL_17;
LABEL_13:
    v30 = &v51[2 * v28];
    ++v18;
    *v30 = (__int64)v25;
    v30[1] = (__int64)v27;
    v19 = v52 + 1;
    LODWORD(v52) = v52 + 1;
    if ( (__int64 *)v20 == v18 )
      goto LABEL_24;
  }
  if ( v51 != (__int64 *)v53 )
    LOBYTE(v5) = _libc_free(v51, a2);
  return (char)v5;
}
