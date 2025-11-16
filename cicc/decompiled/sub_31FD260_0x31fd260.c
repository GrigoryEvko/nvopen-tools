// Function: sub_31FD260
// Address: 0x31fd260
//
unsigned __int64 *__fastcall sub_31FD260(_QWORD *a1, unsigned __int64 a2)
{
  _QWORD *v3; // r13
  _QWORD *v4; // r12
  _QWORD *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  unsigned __int64 v9; // r13
  __int64 v11; // rax
  __int64 v12; // rsi
  _QWORD *v13; // rax
  _QWORD *v14; // rdx
  char v15; // di
  size_t v16; // rdx
  size_t v17; // rbx
  size_t v18; // rdx
  unsigned __int64 *v19; // r15
  size_t v20; // r10
  char v21; // al
  size_t v22; // r10
  unsigned __int64 v23; // rax
  size_t v24; // r10
  __int64 v25; // rbx
  _QWORD *v26; // rax
  unsigned __int64 v27; // r13
  unsigned __int64 v28; // rdx
  __int64 v29; // rax
  char v30; // al
  unsigned __int64 v31; // rdi
  _BYTE *v32; // rax
  unsigned __int64 v33; // rcx
  _BYTE *v34; // rax
  _BYTE *i; // rdx
  unsigned __int64 v36; // r15
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rdx
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // r8
  __int64 v41; // rax
  __int64 v42; // rbx
  unsigned __int64 v43; // rax
  unsigned __int64 v44; // rcx
  unsigned __int64 v45; // rbx
  unsigned __int64 v46; // rcx
  unsigned __int64 v47; // rdx
  unsigned __int64 v48; // rax
  unsigned __int64 *v49; // rax
  unsigned __int64 *v50; // rax
  unsigned __int64 *v51; // rdi
  unsigned __int64 *v52; // rax
  unsigned __int64 *v53; // rdi
  size_t v54; // [rsp+8h] [rbp-C8h]
  size_t v55; // [rsp+8h] [rbp-C8h]
  size_t v56; // [rsp+8h] [rbp-C8h]
  size_t v57; // [rsp+8h] [rbp-C8h]
  size_t n; // [rsp+10h] [rbp-C0h]
  size_t na; // [rsp+10h] [rbp-C0h]
  size_t nb; // [rsp+10h] [rbp-C0h]
  _BYTE *src; // [rsp+18h] [rbp-B8h]
  void *srca; // [rsp+18h] [rbp-B8h]
  void *srcb; // [rsp+18h] [rbp-B8h]
  __int64 v64[2]; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v65; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int64 v66[4]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v67; // [rsp+60h] [rbp-70h]
  unsigned __int64 *v68; // [rsp+70h] [rbp-60h] BYREF
  size_t v69; // [rsp+78h] [rbp-58h]
  _QWORD v70[2]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v71; // [rsp+90h] [rbp-40h]

  v3 = a1 + 175;
  v4 = a1 + 175;
  v6 = (_QWORD *)a1[176];
  if ( !v6 )
    goto LABEL_10;
  do
  {
    while ( 1 )
    {
      v7 = v6[2];
      v8 = v6[3];
      if ( v6[4] >= a2 )
        break;
      v6 = (_QWORD *)v6[3];
      if ( !v8 )
        goto LABEL_6;
    }
    v4 = v6;
    v6 = (_QWORD *)v6[2];
  }
  while ( v7 );
LABEL_6:
  if ( v3 == v4 || v4[4] > a2 )
  {
LABEL_10:
    v11 = sub_22077B0(0x48u);
    v12 = (__int64)v4;
    *(_QWORD *)(v11 + 32) = a2;
    v4 = (_QWORD *)v11;
    *(_QWORD *)(v11 + 40) = v11 + 56;
    *(_QWORD *)(v11 + 48) = 0;
    *(_BYTE *)(v11 + 56) = 0;
    v13 = sub_31FD160(a1 + 174, v12, (unsigned __int64 *)(v11 + 32));
    if ( v14 )
    {
      v15 = v13 || v3 == v14 || a2 < v14[4];
      sub_220F040(v15, (__int64)v4, v14, v3);
      ++a1[179];
      v9 = v4[6];
      if ( v9 )
        return (unsigned __int64 *)v4[5];
      goto LABEL_15;
    }
    v31 = (unsigned __int64)v4;
    v4 = v13;
    j_j___libc_free_0(v31);
  }
  v9 = v4[6];
  if ( v9 )
    return (unsigned __int64 *)v4[5];
LABEL_15:
  src = (_BYTE *)sub_A547D0(a2, 1);
  v17 = v16;
  v19 = (unsigned __int64 *)sub_A547D0(a2, 0);
  v20 = v18;
  if ( v17 && *src == 47 )
  {
    v68 = v19;
    v71 = 261;
    v69 = v18;
    v54 = v18;
    v21 = sub_C81DB0((const char **)&v68, 1);
    v22 = v54;
    if ( !v21 )
    {
      v68 = v70;
      goto LABEL_19;
    }
    return v19;
  }
  if ( v18 )
  {
    if ( *(_BYTE *)v19 == 47 )
    {
      v69 = v18;
      v55 = v18;
      v71 = 261;
      v68 = v19;
      v30 = sub_C81DB0((const char **)&v68, 1);
      v22 = v55;
      if ( !v30 )
      {
        v68 = v70;
        if ( &src[v17] && !src )
          sub_426248((__int64)"basic_string::_M_construct null not valid");
LABEL_19:
        v66[0] = v17;
        if ( v17 > 0xF )
        {
          v57 = v22;
          v50 = (unsigned __int64 *)sub_22409D0((__int64)&v68, v66, 0);
          v22 = v57;
          v68 = v50;
          v51 = v50;
          v70[0] = v66[0];
        }
        else
        {
          if ( v17 == 1 )
          {
            LOBYTE(v70[0]) = *src;
            v23 = 1;
LABEL_22:
            v69 = v23;
            *((_BYTE *)v68 + v23) = 0;
            n = v22;
            sub_2240D70((__int64)(v4 + 5), &v68);
            v24 = n;
            if ( v68 != v70 )
            {
              j_j___libc_free_0((unsigned __int64)v68);
              v24 = n;
            }
            if ( src[v17 - 1] != 47 )
            {
              v25 = v4[6];
              v26 = (_QWORD *)v4[5];
              v27 = v25 + 1;
              if ( v26 == v4 + 7 )
                v28 = 15;
              else
                v28 = v4[7];
              if ( v27 > v28 )
              {
                srcb = (void *)v24;
                sub_2240BB0(v4 + 5, v4[6], 0, 0, 1u);
                v26 = (_QWORD *)v4[5];
                v24 = (size_t)srcb;
              }
              *((_BYTE *)v26 + v25) = 47;
              v29 = v4[5];
              v4[6] = v27;
              *(_BYTE *)(v29 + v25 + 1) = 0;
            }
            if ( v24 > 0x3FFFFFFFFFFFFFFFLL - v4[6] )
              sub_4262D8((__int64)"basic_string::append");
            sub_2241490(v4 + 5, (char *)v19, v24);
            return (unsigned __int64 *)v4[5];
          }
          if ( !v17 )
          {
            v23 = 0;
            goto LABEL_22;
          }
          v51 = v70;
        }
        nb = v22;
        memcpy(v51, src, v17);
        v23 = v66[0];
        v22 = nb;
        goto LABEL_22;
      }
      return v19;
    }
    na = v18;
    v56 = v18;
    v32 = memchr(v19, 58, v18);
    v20 = na;
    if ( v32 )
    {
      v33 = v56;
      if ( v32 - (_BYTE *)v19 == 1 )
      {
        v66[0] = na;
        v68 = v70;
        if ( na > 0xF )
        {
          v52 = (unsigned __int64 *)sub_22409D0((__int64)&v68, v66, 0);
          v20 = na;
          v68 = v52;
          v53 = v52;
          v70[0] = v66[0];
        }
        else
        {
          if ( na == 1 )
          {
            LOBYTE(v70[0]) = *(_BYTE *)v19;
            v49 = v70;
            goto LABEL_73;
          }
          v53 = v70;
        }
        memcpy(v53, v19, v20);
        v33 = v66[0];
        v49 = v68;
LABEL_73:
        v69 = v33;
        *((_BYTE *)v49 + v33) = 0;
        sub_2240D70((__int64)(v4 + 5), &v68);
        if ( v68 != v70 )
          j_j___libc_free_0((unsigned __int64)v68);
        goto LABEL_41;
      }
    }
  }
  v70[0] = v19;
  v67 = 773;
  v70[1] = v20;
  v66[0] = (unsigned __int64)src;
  v66[2] = (unsigned __int64)"\\";
  v68 = v66;
  v71 = 1282;
  v66[1] = v17;
  sub_CA0F50(v64, (void **)&v68);
  sub_2240D70((__int64)(v4 + 5), v64);
  if ( (__int64 *)v64[0] != &v65 )
    j_j___libc_free_0(v64[0]);
LABEL_41:
  v34 = (_BYTE *)v4[5];
  for ( i = &v34[v4[6]]; i != v34; ++v34 )
  {
    if ( *v34 == 47 )
      *v34 = 92;
  }
  v36 = 0;
  while ( 1 )
  {
    v39 = sub_22416F0(v4 + 5, "\\.\\", v36, 3u);
    v36 = v39;
    if ( v39 == -1 )
      break;
    v37 = v4[6];
    if ( v39 > v37 )
      sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", "basic_string::erase", v39, v4[6]);
    v38 = v37 - v39;
    if ( v38 > 2 )
      v38 = 2;
    sub_2240CE0(v4 + 5, v39, v38);
  }
  v40 = 0;
  while ( 1 )
  {
    v41 = sub_22416F0(v4 + 5, "\\..\\", v40, 4u);
    v42 = v41;
    if ( (unsigned __int64)(v41 - 1) > 0xFFFFFFFFFFFFFFFDLL )
      break;
    v43 = sub_22418A0(v4 + 5, 92, v41 - 1);
    v40 = v43;
    if ( v43 == -1 )
      break;
    v44 = v4[6];
    v45 = v42 + 3 - v43;
    if ( v43 > v44 )
      sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", "basic_string::erase", v43, v44);
    if ( v45 == -1 )
    {
      v4[6] = v43;
      *(_BYTE *)(v4[5] + v43) = 0;
    }
    else if ( v45 )
    {
      v46 = v44 - v43;
      srca = (void *)v43;
      if ( v46 <= v45 )
        v45 = v46;
      sub_2240CE0(v4 + 5, v43, v45);
      v40 = (unsigned __int64)srca;
    }
  }
  while ( 1 )
  {
    v48 = sub_22416F0(v4 + 5, "\\\\", v9, 2u);
    v9 = v48;
    if ( v48 == -1 )
      break;
    v47 = v4[6];
    if ( v48 > v47 )
      sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", "basic_string::erase", v48, v4[6]);
    sub_2240CE0(v4 + 5, v48, v47 != v48);
  }
  return (unsigned __int64 *)v4[5];
}
