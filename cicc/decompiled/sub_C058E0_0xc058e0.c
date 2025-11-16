// Function: sub_C058E0
// Address: 0xc058e0
//
unsigned __int64 __fastcall sub_C058E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 *v5; // r12
  unsigned __int64 result; // rax
  __int64 v7; // rsi
  __int64 v8; // r14
  char *v10; // rax
  __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r15
  _QWORD *v19; // rax
  char *v20; // rdx
  __int64 v21; // rdx
  _BYTE *v22; // rcx
  __int64 v23; // rax
  __int64 *v24; // rax
  __int64 v25; // r9
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // r14
  _BYTE **v29; // r13
  __int64 v30; // r15
  _BYTE *v31; // rax
  _BYTE **v32; // r14
  unsigned __int64 v33; // rdi
  _BYTE *v34; // rdi
  char *v35; // rax
  unsigned int v36; // eax
  __int64 v37; // rax
  __int64 v38; // r9
  __int64 v39; // rax
  _QWORD *v40; // rax
  __int64 *v41; // rdx
  int v42; // eax
  int v43; // r10d
  __int64 *v44; // [rsp+0h] [rbp-190h]
  __int64 v45; // [rsp+0h] [rbp-190h]
  __int64 v46; // [rsp+0h] [rbp-190h]
  __int64 v47; // [rsp+8h] [rbp-188h]
  __int64 v48; // [rsp+18h] [rbp-178h] BYREF
  const char *v49; // [rsp+20h] [rbp-170h] BYREF
  char v50; // [rsp+40h] [rbp-150h]
  char v51; // [rsp+41h] [rbp-14Fh]
  _BYTE *v52; // [rsp+50h] [rbp-140h] BYREF
  __int64 v53; // [rsp+58h] [rbp-138h]
  _BYTE v54[64]; // [rsp+60h] [rbp-130h] BYREF
  __int64 v55; // [rsp+A0h] [rbp-F0h] BYREF
  char *v56; // [rsp+A8h] [rbp-E8h]
  __int64 v57; // [rsp+B0h] [rbp-E0h]
  int v58; // [rsp+B8h] [rbp-D8h]
  unsigned __int8 v59; // [rsp+BCh] [rbp-D4h]
  char v60; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v61; // [rsp+100h] [rbp-90h] BYREF
  void *s; // [rsp+108h] [rbp-88h]
  _BYTE v63[12]; // [rsp+110h] [rbp-80h]
  unsigned __int8 v64; // [rsp+11Ch] [rbp-74h]
  _BYTE v65[112]; // [rsp+120h] [rbp-70h] BYREF

  v4 = *(unsigned int *)(a1 + 904);
  v5 = *(__int64 **)(a1 + 896);
  v56 = &v60;
  result = (unsigned __int64)v65;
  v64 = 1;
  v4 *= 16;
  v55 = 0;
  v7 = (__int64)v5 + v4;
  v59 = 1;
  v57 = 8;
  v58 = 0;
  v61 = 0;
  s = v65;
  *(_QWORD *)v63 = 8;
  *(_DWORD *)&v63[8] = 0;
  v44 = (__int64 *)((char *)v5 + v4);
  if ( (__int64 *)((char *)v5 + v4) == v5 )
    return result;
  v8 = *v5;
LABEL_3:
  v10 = v56;
  v11 = (__int64)&v56[8 * HIDWORD(v57)];
  if ( v56 != (char *)v11 )
  {
    do
    {
      if ( v8 == *(_QWORD *)v10 )
        goto LABEL_7;
      v10 += 8;
    }
    while ( (char *)v11 != v10 );
  }
LABEL_10:
  if ( !v64 )
    goto LABEL_78;
  v12 = s;
  v13 = (__int64 *)((char *)s + 8 * *(unsigned int *)&v63[4]);
  if ( s != v13 )
  {
    while ( v8 != *v12 )
    {
      if ( v13 == ++v12 )
        goto LABEL_77;
    }
    goto LABEL_15;
  }
LABEL_77:
  if ( *(_DWORD *)&v63[4] < *(_DWORD *)v63 )
  {
    ++*(_DWORD *)&v63[4];
    *v13 = v8;
    ++v61;
  }
  else
  {
LABEL_78:
    v7 = v8;
    sub_C8CC70(&v61, v8);
  }
LABEL_15:
  v14 = v5[1];
  while ( 1 )
  {
    v15 = sub_BD99E0(v14);
    v18 = v15;
    if ( v64 )
    {
      v19 = s;
      v20 = (char *)s + 8 * *(unsigned int *)&v63[4];
      if ( s == v20 )
        goto LABEL_44;
      while ( v18 != *v19 )
      {
        if ( v20 == (char *)++v19 )
          goto LABEL_44;
      }
LABEL_21:
      v48 = v18;
      v21 = 0;
      v52 = v54;
      v22 = v54;
      v53 = 0x800000000LL;
      v23 = v18;
      while ( 1 )
      {
        *(_QWORD *)&v22[8 * v21] = v23;
        v7 = (__int64)&v48;
        LODWORD(v53) = v53 + 1;
        v24 = (__int64 *)sub_C04EB0(a1 + 864, &v48);
        v25 = *v24;
        if ( v48 != *v24 )
        {
          v26 = (unsigned int)v53;
          v27 = (unsigned int)v53 + 1LL;
          if ( v27 > HIDWORD(v53) )
          {
            v7 = (__int64)v54;
            v46 = v25;
            sub_C8D5F0(&v52, v54, v27, 8);
            v26 = (unsigned int)v53;
            v25 = v46;
          }
          *(_QWORD *)&v52[8 * v26] = v25;
          LODWORD(v53) = v53 + 1;
        }
        v23 = sub_BD99E0(v25);
        v48 = v23;
        if ( v23 == v18 )
          break;
        v21 = (unsigned int)v53;
        if ( (unsigned __int64)(unsigned int)v53 + 1 > HIDWORD(v53) )
        {
          v45 = v23;
          sub_C8D5F0(&v52, v54, (unsigned int)v53 + 1LL, 8);
          v21 = (unsigned int)v53;
          v23 = v45;
        }
        v22 = v52;
      }
      v28 = *(_QWORD *)a1;
      result = (unsigned __int64)"EH pads can't handle each other's exceptions";
      v51 = 1;
      v49 = "EH pads can't handle each other's exceptions";
      v29 = (_BYTE **)v52;
      v50 = 3;
      v30 = (unsigned int)v53;
      if ( !v28 )
      {
        *(_BYTE *)(a1 + 152) = 1;
        goto LABEL_81;
      }
      v7 = v28;
      sub_CA0E80(&v49, v28);
      v31 = *(_BYTE **)(v28 + 32);
      if ( (unsigned __int64)v31 >= *(_QWORD *)(v28 + 24) )
      {
        v7 = 10;
        sub_CB5D20(v28, 10);
      }
      else
      {
        *(_QWORD *)(v28 + 32) = v31 + 1;
        *v31 = 10;
      }
      result = *(_QWORD *)a1;
      *(_BYTE *)(a1 + 152) = 1;
      if ( !result || (v32 = &v29[v30], v29 == v32) )
      {
LABEL_81:
        if ( v52 != v54 )
          result = _libc_free(v52, v7);
        if ( !v64 )
          result = _libc_free(s, v7);
        if ( v59 )
          return result;
        return _libc_free(v56, v7);
      }
      while ( 1 )
      {
        v34 = *v29;
        if ( *v29 )
        {
          v7 = *(_QWORD *)a1;
          if ( *v34 > 0x1Cu )
          {
            sub_A693B0((__int64)v34, (_BYTE *)v7, a1 + 16, 0);
            v33 = *(_QWORD *)a1;
            result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
            if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
              goto LABEL_42;
          }
          else
          {
            sub_A5C020(v34, v7, 1, a1 + 16);
            v33 = *(_QWORD *)a1;
            result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
            if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
            {
LABEL_42:
              v7 = 10;
              result = sub_CB5D20(v33, 10);
              goto LABEL_38;
            }
          }
          *(_QWORD *)(v33 + 32) = result + 1;
          *(_BYTE *)result = 10;
        }
LABEL_38:
        if ( v32 == ++v29 )
          goto LABEL_81;
      }
    }
    v7 = v15;
    if ( sub_C8CA60(&v61, v15, v16, v17) )
      goto LABEL_21;
LABEL_44:
    if ( !v59 )
      goto LABEL_59;
    v35 = v56;
    a4 = HIDWORD(v57);
    v11 = (__int64)&v56[8 * HIDWORD(v57)];
    if ( v56 != (char *)v11 )
      break;
LABEL_71:
    if ( HIDWORD(v57) >= (unsigned int)v57 )
    {
LABEL_59:
      v7 = v18;
      sub_C8CC70(&v55, v18);
      if ( !(_BYTE)v11 )
        goto LABEL_49;
      goto LABEL_60;
    }
    a4 = (unsigned int)++HIDWORD(v57);
    *(_QWORD *)v11 = v18;
    ++v55;
LABEL_60:
    v11 = *(unsigned int *)(a1 + 888);
    v7 = *(_QWORD *)(a1 + 872);
    if ( !(_DWORD)v11 )
      goto LABEL_49;
    a4 = ((_DWORD)v11 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
    v37 = v7 + 16 * a4;
    v38 = *(_QWORD *)v37;
    if ( v18 != *(_QWORD *)v37 )
    {
      v42 = 1;
      while ( v38 != -4096 )
      {
        v43 = v42 + 1;
        a4 = ((_DWORD)v11 - 1) & (unsigned int)(v42 + a4);
        v37 = v7 + 16LL * (unsigned int)a4;
        v38 = *(_QWORD *)v37;
        if ( v18 == *(_QWORD *)v37 )
          goto LABEL_62;
        v42 = v43;
      }
      goto LABEL_49;
    }
LABEL_62:
    a4 = v64;
    v11 = v7 + 16 * v11;
    if ( v37 == v11 )
      goto LABEL_49;
    v39 = *(_QWORD *)(a1 + 896) + 16LL * *(unsigned int *)(v37 + 8);
    v11 = *(_QWORD *)(a1 + 896) + 16LL * *(unsigned int *)(a1 + 904);
    if ( v39 == v11 )
      goto LABEL_49;
    v14 = *(_QWORD *)(v39 + 8);
    if ( v64 )
    {
      v40 = s;
      v41 = (__int64 *)((char *)s + 8 * *(unsigned int *)&v63[4]);
      if ( s == v41 )
      {
LABEL_68:
        if ( *(_DWORD *)&v63[4] >= *(_DWORD *)v63 )
          goto LABEL_70;
        ++*(_DWORD *)&v63[4];
        *v41 = v18;
        ++v61;
      }
      else
      {
        while ( v18 != *v40 )
        {
          if ( v41 == ++v40 )
            goto LABEL_68;
        }
      }
    }
    else
    {
LABEL_70:
      v7 = v18;
      v47 = v14;
      sub_C8CC70(&v61, v18);
      v14 = v47;
    }
  }
  while ( v18 != *(_QWORD *)v35 )
  {
    v35 += 8;
    if ( (char *)v11 == v35 )
      goto LABEL_71;
  }
LABEL_49:
  ++v61;
  if ( v64 )
  {
LABEL_54:
    *(_QWORD *)&v63[4] = 0;
    v5 += 2;
    if ( v44 != v5 )
      goto LABEL_8;
  }
  else
  {
    v36 = 4 * (*(_DWORD *)&v63[4] - *(_DWORD *)&v63[8]);
    if ( v36 < 0x20 )
      v36 = 32;
    if ( *(_DWORD *)v63 <= v36 )
    {
      v7 = 0xFFFFFFFFLL;
      memset(s, -1, 8LL * *(unsigned int *)v63);
      goto LABEL_54;
    }
    sub_C8C990(&v61);
LABEL_7:
    while ( 1 )
    {
      v5 += 2;
      if ( v44 == v5 )
        break;
LABEL_8:
      v8 = *v5;
      if ( v59 )
        goto LABEL_3;
      v7 = *v5;
      if ( !sub_C8CA60(&v55, v8, v11, a4) )
        goto LABEL_10;
    }
  }
  if ( !v64 )
    _libc_free(s, v7);
  result = v59;
  if ( !v59 )
    return _libc_free(v56, v7);
  return result;
}
