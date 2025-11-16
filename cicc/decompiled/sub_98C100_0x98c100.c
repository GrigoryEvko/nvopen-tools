// Function: sub_98C100
// Address: 0x98c100
//
unsigned __int8 *__fastcall sub_98C100(__int64 a1, char a2)
{
  unsigned __int8 *v2; // r12
  __int64 v3; // rsi
  _BYTE *v4; // rcx
  unsigned int v5; // eax
  _BYTE *v6; // rdi
  __int64 v7; // rdx
  unsigned __int8 *v8; // r13
  int v9; // edx
  __int64 *v10; // rdi
  __int64 v11; // r15
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 *v18; // rdi
  __int64 *v19; // rax
  __int64 v20; // rcx
  __int64 *v21; // rdx
  char v22; // dl
  _BYTE **v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdx
  unsigned __int8 *v26; // r10
  unsigned __int8 *v27; // r15
  __int64 *v28; // r13
  __int64 *v29; // rdi
  __int64 v30; // r14
  __int64 *v31; // rax
  __int64 v32; // rcx
  __int64 *v33; // rdx
  char v34; // dl
  _BYTE **v35; // rax
  __int64 v36; // rdx
  char v37; // dl
  _BYTE **v38; // r13
  __int64 v39; // rdx
  _BYTE **v40; // [rsp+8h] [rbp-C8h]
  _BYTE **v41; // [rsp+8h] [rbp-C8h]
  __int64 *v43; // [rsp+20h] [rbp-B0h] BYREF
  _BYTE **v44; // [rsp+28h] [rbp-A8h]
  _BYTE *v45; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v46; // [rsp+38h] [rbp-98h]
  _BYTE v47[32]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v48; // [rsp+60h] [rbp-70h] BYREF
  char *v49; // [rsp+68h] [rbp-68h]
  __int64 v50; // [rsp+70h] [rbp-60h]
  int v51; // [rsp+78h] [rbp-58h]
  char v52; // [rsp+7Ch] [rbp-54h]
  char v53; // [rsp+80h] [rbp-50h] BYREF

  v2 = 0;
  v45 = v47;
  v46 = 0x400000000LL;
  v3 = a1;
  v49 = &v53;
  v44 = &v45;
  v48 = 0;
  v50 = 4;
  v51 = 0;
  v52 = 1;
  v43 = &v48;
  sub_984BF0((__int64 *)&v43, a1);
  v4 = v45;
  v5 = v46;
  v6 = v45;
  while ( 1 )
  {
    v7 = v5--;
    v8 = *(unsigned __int8 **)&v4[8 * v7 - 8];
    LODWORD(v46) = v5;
    v9 = *v8;
    if ( (unsigned __int8)v9 <= 0x1Cu )
    {
LABEL_17:
      v2 = 0;
      goto LABEL_12;
    }
    if ( (_BYTE)v9 != 60 )
      break;
    if ( v2 && v2 != v8 )
      goto LABEL_17;
    v2 = v8;
LABEL_11:
    if ( !v5 )
      goto LABEL_12;
  }
  v3 = (unsigned int)(unsigned __int8)v9 - 67;
  if ( (unsigned int)v3 <= 0xC )
  {
    v10 = v43;
    v11 = *((_QWORD *)v8 - 4);
    if ( *((_BYTE *)v43 + 28) )
    {
      v12 = (_QWORD *)v43[1];
      v3 = *((unsigned int *)v43 + 5);
      v13 = &v12[v3];
      if ( v12 != v13 )
      {
        while ( v11 != *v12 )
        {
          if ( v13 == ++v12 )
            goto LABEL_63;
        }
LABEL_10:
        v5 = v46;
        v6 = v4;
        goto LABEL_11;
      }
LABEL_63:
      if ( (unsigned int)v3 < *((_DWORD *)v43 + 4) )
      {
        v3 = (unsigned int)(v3 + 1);
        *((_DWORD *)v43 + 5) = v3;
        *v13 = v11;
        ++*v10;
LABEL_65:
        v38 = v44;
        v39 = *((unsigned int *)v44 + 2);
        if ( v39 + 1 > (unsigned __int64)*((unsigned int *)v44 + 3) )
        {
          v3 = (__int64)(v44 + 2);
          sub_C8D5F0(v44, v44 + 2, v39 + 1, 8);
          v39 = *((unsigned int *)v38 + 2);
        }
        *(_QWORD *)&(*v38)[8 * v39] = v11;
        v4 = v45;
        ++*((_DWORD *)v38 + 2);
        v5 = v46;
        v6 = v4;
        goto LABEL_11;
      }
    }
    v3 = *((_QWORD *)v8 - 4);
    sub_C8CC70(v43, v3);
    if ( v37 )
      goto LABEL_65;
    v4 = v45;
    goto LABEL_10;
  }
  if ( (_BYTE)v9 == 84 )
  {
    v25 = 32LL * (*((_DWORD *)v8 + 1) & 0x7FFFFFF);
    if ( (v8[7] & 0x40) != 0 )
    {
      v26 = (unsigned __int8 *)*((_QWORD *)v8 - 1);
      v27 = &v26[v25];
    }
    else
    {
      v27 = v8;
      v26 = &v8[-v25];
    }
    if ( v27 == v26 )
      goto LABEL_11;
    v28 = (__int64 *)v26;
    while ( 1 )
    {
      v29 = v43;
      v30 = *v28;
      if ( *((_BYTE *)v43 + 28) )
      {
        v31 = (__int64 *)v43[1];
        v32 = *((unsigned int *)v43 + 5);
        v33 = &v31[v32];
        if ( v31 != v33 )
        {
          while ( v30 != *v31 )
          {
            if ( v33 == ++v31 )
              goto LABEL_55;
          }
          goto LABEL_48;
        }
LABEL_55:
        if ( (unsigned int)v32 < *((_DWORD *)v43 + 4) )
        {
          *((_DWORD *)v43 + 5) = v32 + 1;
          *v33 = v30;
          ++*v29;
          goto LABEL_51;
        }
      }
      v3 = *v28;
      sub_C8CC70(v43, *v28);
      if ( v34 )
      {
LABEL_51:
        v35 = v44;
        v36 = *((unsigned int *)v44 + 2);
        if ( v36 + 1 > (unsigned __int64)*((unsigned int *)v44 + 3) )
        {
          v3 = (__int64)(v44 + 2);
          v40 = v44;
          sub_C8D5F0(v44, v44 + 2, v36 + 1, 8);
          v35 = v40;
          v36 = *((unsigned int *)v40 + 2);
        }
        v28 += 4;
        *(_QWORD *)&(*v35)[8 * v36] = v30;
        ++*((_DWORD *)v35 + 2);
        if ( v27 == (unsigned __int8 *)v28 )
          goto LABEL_29;
      }
      else
      {
LABEL_48:
        v28 += 4;
        if ( v27 == (unsigned __int8 *)v28 )
          goto LABEL_29;
      }
    }
  }
  if ( (_BYTE)v9 == 86 )
  {
    sub_984BF0((__int64 *)&v43, *((_QWORD *)v8 - 8));
    v3 = *((_QWORD *)v8 - 4);
LABEL_60:
    sub_984BF0((__int64 *)&v43, v3);
    v4 = v45;
    v5 = v46;
    v6 = v45;
    goto LABEL_11;
  }
  if ( (_BYTE)v9 != 63 )
  {
    v15 = (unsigned int)(v9 - 34);
    if ( (unsigned __int8)v15 > 0x33u )
      goto LABEL_17;
    v16 = 0x8000000000041LL;
    if ( !_bittest64(&v16, v15) )
      goto LABEL_17;
    v3 = 52;
    v17 = sub_B494D0(v8, 52);
    if ( !v17 )
      goto LABEL_70;
    v18 = v43;
    if ( !*((_BYTE *)v43 + 28) )
      goto LABEL_30;
    v19 = (__int64 *)v43[1];
    v20 = *((unsigned int *)v43 + 5);
    v21 = &v19[v20];
    if ( v19 == v21 )
    {
LABEL_37:
      if ( (unsigned int)v20 < *((_DWORD *)v43 + 4) )
      {
        *((_DWORD *)v43 + 5) = v20 + 1;
        *v21 = v17;
        ++*v18;
LABEL_31:
        v23 = v44;
        v24 = *((unsigned int *)v44 + 2);
        if ( v24 + 1 > (unsigned __int64)*((unsigned int *)v44 + 3) )
        {
          v3 = (__int64)(v44 + 2);
          v41 = v44;
          sub_C8D5F0(v44, v44 + 2, v24 + 1, 8);
          v23 = v41;
          v24 = *((unsigned int *)v41 + 2);
        }
        *(_QWORD *)&(*v23)[8 * v24] = v17;
        v4 = v45;
        ++*((_DWORD *)v23 + 2);
        v5 = v46;
        v6 = v4;
        goto LABEL_11;
      }
LABEL_30:
      v3 = v17;
      sub_C8CC70(v43, v17);
      if ( v22 )
        goto LABEL_31;
    }
    else
    {
      while ( v17 != *v19 )
      {
        if ( v21 == ++v19 )
          goto LABEL_37;
      }
    }
LABEL_29:
    v4 = v45;
    v5 = v46;
    v6 = v45;
    goto LABEL_11;
  }
  if ( !a2 || (unsigned __int8)sub_B4DCF0(v8) )
  {
    v3 = *(_QWORD *)&v8[-32 * (*((_DWORD *)v8 + 1) & 0x7FFFFFF)];
    goto LABEL_60;
  }
LABEL_70:
  v6 = v45;
  v2 = 0;
LABEL_12:
  if ( v6 != v47 )
    _libc_free(v6, v3);
  if ( !v52 )
    _libc_free(v49, v3);
  return v2;
}
