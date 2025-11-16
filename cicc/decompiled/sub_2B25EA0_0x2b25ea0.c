// Function: sub_2B25EA0
// Address: 0x2b25ea0
//
__int64 __fastcall sub_2B25EA0(_BYTE **a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r14
  _QWORD *v9; // rax
  unsigned int v10; // esi
  __int64 v11; // rdx
  unsigned int v12; // edx
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rax
  _QWORD *v16; // rbx
  _QWORD *v17; // r15
  _BYTE *v18; // rax
  _BYTE *v19; // rax
  _BYTE *v20; // rax
  unsigned __int8 *v21; // rdi
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rdx
  size_t v24; // rdx
  void *v25; // rdi
  unsigned __int64 v26; // r12
  int v27; // r13d
  unsigned __int8 *v28; // r15
  unsigned __int8 *v29; // rbx
  __int64 v30; // rsi
  unsigned __int64 v31; // r9
  char v32; // cl
  unsigned __int64 v33; // r9
  unsigned __int64 v34; // r14
  char v35; // dl
  unsigned __int64 *v36; // r14
  unsigned __int8 *v37; // rax
  _BOOL4 v38; // eax
  unsigned __int8 *v40; // rdi
  unsigned __int8 *v41; // rdi
  unsigned __int8 *v42; // rdi
  unsigned int v43; // esi
  unsigned int v44; // edi
  _QWORD *v45; // rax
  unsigned __int8 *v46; // rdx
  int v47; // ecx
  unsigned __int64 v48; // rax
  int v49; // esi
  unsigned int v50; // ecx
  unsigned int v51; // edi
  _QWORD *v52; // rax
  int v53; // ecx
  int v54; // eax
  unsigned __int8 *v55; // rdi
  unsigned __int8 *v56; // rdi
  unsigned __int8 *v57; // rdi
  char v58; // [rsp+8h] [rbp-98h]
  char v59; // [rsp+8h] [rbp-98h]
  __int64 v60; // [rsp+8h] [rbp-98h]
  _BYTE **v61; // [rsp+10h] [rbp-90h]
  unsigned __int8 *v62; // [rsp+18h] [rbp-88h]
  unsigned __int64 v63; // [rsp+20h] [rbp-80h]
  char v64; // [rsp+20h] [rbp-80h]
  unsigned __int64 v65; // [rsp+20h] [rbp-80h]
  char v66; // [rsp+20h] [rbp-80h]
  int v67; // [rsp+20h] [rbp-80h]
  char v68; // [rsp+20h] [rbp-80h]
  char v69; // [rsp+20h] [rbp-80h]
  char v70; // [rsp+20h] [rbp-80h]
  char v71; // [rsp+20h] [rbp-80h]
  unsigned int v72; // [rsp+2Ch] [rbp-74h]
  unsigned __int8 *v73; // [rsp+30h] [rbp-70h]
  _BYTE **v74; // [rsp+38h] [rbp-68h]
  int v76; // [rsp+50h] [rbp-50h]
  int v77; // [rsp+54h] [rbp-4Ch]
  unsigned __int64 v79; // [rsp+60h] [rbp-40h] BYREF
  _QWORD v80[7]; // [rsp+68h] [rbp-38h] BYREF

  v3 = 8 * a2;
  v74 = &a1[a2];
  if ( v74 == sub_2B0CAD0(a1, (__int64)v74) )
  {
LABEL_57:
    BYTE4(v80[0]) = 0;
    return v80[0];
  }
  v8 = v5;
  v9 = a1;
  if ( a1 == v74 )
  {
    v72 = 0;
  }
  else
  {
    v10 = 0;
    do
    {
      while ( 1 )
      {
        if ( *(_BYTE *)*v9 == 90 )
        {
          v11 = *(_QWORD *)(*(_QWORD *)(*v9 - 64LL) + 8LL);
          if ( *(_BYTE *)(v11 + 8) == 17 )
            break;
        }
        if ( ++v9 == v74 )
          goto LABEL_10;
      }
      v12 = *(_DWORD *)(v11 + 32);
      if ( v10 < v12 )
        v10 = v12;
      ++v9;
    }
    while ( v9 != v74 );
LABEL_10:
    v72 = v10;
  }
  v13 = v3;
  v14 = v3 >> 5;
  v15 = v13 >> 3;
  if ( v14 <= 0 )
  {
    v16 = a1;
LABEL_115:
    if ( v15 != 2 )
    {
      if ( v15 != 3 )
      {
        if ( v15 != 1 )
          goto LABEL_118;
        goto LABEL_124;
      }
      if ( *(_BYTE *)*v16 == 90 )
      {
        v57 = *(unsigned __int8 **)(*v16 - 64LL);
        if ( (unsigned int)*v57 - 12 > 1 )
        {
          if ( sub_98ED70(v57, v8, 0, 0, 0) )
            goto LABEL_20;
        }
      }
      ++v16;
    }
    if ( *(_BYTE *)*v16 == 90 )
    {
      v56 = *(unsigned __int8 **)(*v16 - 64LL);
      if ( (unsigned int)*v56 - 12 > 1 )
      {
        if ( sub_98ED70(v56, v8, 0, 0, 0) )
          goto LABEL_20;
      }
    }
    ++v16;
LABEL_124:
    if ( *(_BYTE *)*v16 == 90 )
    {
      v55 = *(unsigned __int8 **)(*v16 - 64LL);
      if ( (unsigned int)*v55 - 12 > 1 )
      {
        if ( !sub_98ED70(v55, v8, 0, 0, 0) )
          v16 = v74;
        goto LABEL_20;
      }
    }
LABEL_118:
    v16 = v74;
    goto LABEL_20;
  }
  v16 = a1;
  v17 = &a1[4 * v14];
  while ( 1 )
  {
    if ( *(_BYTE *)*v16 == 90 )
    {
      v21 = *(unsigned __int8 **)(*v16 - 64LL);
      if ( (unsigned int)*v21 - 12 > 1 )
      {
        if ( sub_98ED70(v21, v8, 0, 0, 0) )
          break;
      }
    }
    v18 = (_BYTE *)v16[1];
    if ( *v18 == 90 )
    {
      v40 = (unsigned __int8 *)*((_QWORD *)v18 - 8);
      if ( (unsigned int)*v40 - 12 > 1 )
      {
        if ( sub_98ED70(v40, v8, 0, 0, 0) )
        {
          ++v16;
          break;
        }
      }
    }
    v19 = (_BYTE *)v16[2];
    if ( *v19 == 90 )
    {
      v41 = (unsigned __int8 *)*((_QWORD *)v19 - 8);
      if ( (unsigned int)*v41 - 12 > 1 )
      {
        if ( sub_98ED70(v41, v8, 0, 0, 0) )
        {
          v16 += 2;
          break;
        }
      }
    }
    v20 = (_BYTE *)v16[3];
    if ( *v20 == 90 )
    {
      v42 = (unsigned __int8 *)*((_QWORD *)v20 - 8);
      if ( (unsigned int)*v42 - 12 > 1 )
      {
        if ( sub_98ED70(v42, v8, 0, 0, 0) )
        {
          v16 += 3;
          break;
        }
      }
    }
    v16 += 4;
    if ( v17 == v16 )
    {
      v15 = v74 - (_BYTE **)v16;
      goto LABEL_115;
    }
  }
LABEL_20:
  if ( a2 > *(unsigned int *)(a3 + 12) )
  {
    *(_DWORD *)(a3 + 8) = 0;
    sub_C8D5F0(a3, (const void *)(a3 + 16), a2, 4u, v6, v7);
    v25 = *(void **)a3;
    v24 = 4 * a2;
    if ( !(4 * a2) )
      goto LABEL_28;
  }
  else
  {
    v22 = *(unsigned int *)(a3 + 8);
    v23 = v22;
    if ( a2 <= v22 )
      v23 = a2;
    if ( v23 )
    {
      memset(*(void **)a3, 255, 4 * v23);
      v22 = *(unsigned int *)(a3 + 8);
    }
    if ( a2 <= v22 )
      goto LABEL_28;
    v24 = 4 * (a2 - v22);
    v25 = (void *)(*(_QWORD *)a3 + 4 * v22);
    if ( !v24 )
      goto LABEL_28;
  }
  memset(v25, 255, v24);
LABEL_28:
  v77 = a2;
  *(_DWORD *)(a3 + 8) = a2;
  if ( !(_DWORD)a2 )
    goto LABEL_110;
  v76 = 0;
  v26 = 0;
  v62 = 0;
  v27 = 0;
  v73 = 0;
  v61 = (_BYTE **)v16;
  do
  {
    while ( 1 )
    {
      v28 = a1[v26 / 4];
      if ( (unsigned int)*v28 - 12 <= 1 )
        goto LABEL_30;
      v29 = (unsigned __int8 *)*((_QWORD *)v28 - 8);
      if ( *(_BYTE *)(*((_QWORD *)v29 + 1) + 8LL) == 18 )
        goto LABEL_57;
      v30 = *((_QWORD *)v28 - 8);
      v79 = 1;
      sub_2B25530(v80, v30, &v79);
      v31 = v80[0];
      v32 = v80[0] & 1;
      if ( (v80[0] & 1) != 0 )
      {
        v32 = (~(-1LL << (v80[0] >> 58)) & (v80[0] >> 1)) == (1LL << (v80[0] >> 58)) - 1;
      }
      else
      {
        v43 = *(_DWORD *)(v80[0] + 64LL);
        v44 = v43 >> 6;
        if ( v43 >> 6 )
        {
          v45 = *(_QWORD **)v80[0];
          while ( *v45 == -1 )
          {
            if ( ++v45 == (_QWORD *)(*(_QWORD *)v80[0] + 8LL * (v44 - 1) + 8) )
              goto LABEL_82;
          }
        }
        else
        {
LABEL_82:
          v32 = 1;
          v49 = v43 & 0x3F;
          if ( v49 )
            v32 = *(_QWORD *)(*(_QWORD *)v80[0] + 8LL * v44) == (1LL << v49) - 1;
        }
        if ( v80[0] )
        {
          if ( *(_QWORD *)v80[0] != v80[0] + 16LL )
          {
            v59 = v32;
            v65 = v80[0];
            _libc_free(*(_QWORD *)v80[0]);
            v32 = v59;
            v31 = v65;
          }
          v66 = v32;
          j_j___libc_free_0(v31);
          v32 = v66;
        }
      }
      v33 = v79;
      if ( (v79 & 1) == 0 && v79 )
      {
        if ( *(_QWORD *)v79 != v79 + 16 )
        {
          v58 = v32;
          v63 = v79;
          _libc_free(*(_QWORD *)v79);
          v32 = v58;
          v33 = v63;
        }
        v64 = v32;
        j_j___libc_free_0(v33);
        v32 = v64;
      }
      if ( v32 )
        goto LABEL_30;
      if ( (unsigned int)*v29 - 12 > 1 )
      {
        v46 = (unsigned __int8 *)*((_QWORD *)v28 - 4);
        v47 = *v46;
        if ( (unsigned int)(v47 - 12) <= 1 )
          goto LABEL_30;
        if ( (_BYTE)v47 != 17 )
          goto LABEL_57;
        if ( *((_DWORD *)v46 + 8) <= 0x40u )
        {
          v48 = *((_QWORD *)v46 + 3);
          if ( v72 <= v48 )
            goto LABEL_30;
        }
        else
        {
          v60 = *((_QWORD *)v28 - 4);
          v67 = *((_DWORD *)v46 + 8);
          if ( v67 - (unsigned int)sub_C444A0((__int64)(v46 + 24)) > 0x40 )
            goto LABEL_30;
          v48 = **(_QWORD **)(v60 + 24);
          if ( v72 <= v48 )
            goto LABEL_30;
        }
        *(_DWORD *)(*(_QWORD *)a3 + v26) = v48;
      }
      else
      {
        *(_DWORD *)(*(_QWORD *)a3 + v26) = v27;
      }
      v79 = 1;
      sub_2B25A00(v80, (char *)v29, &v79);
      v34 = v80[0];
      v35 = v80[0] & 1;
      if ( (v80[0] & 1) != 0 )
      {
        if ( (~(-1LL << (v80[0] >> 58)) & (v80[0] >> 1)) != (1LL << (v80[0] >> 58)) - 1 || v74 == v61 )
        {
          v36 = (unsigned __int64 *)v79;
          if ( (v79 & 1) != 0 || !v79 )
          {
            v37 = v73;
            if ( !v73 )
              goto LABEL_99;
            goto LABEL_48;
          }
          v35 = 0;
        }
        else
        {
          v36 = (unsigned __int64 *)v79;
          if ( (v79 & 1) != 0 || !v79 )
            goto LABEL_30;
        }
LABEL_94:
        if ( (unsigned __int64 *)*v36 != v36 + 2 )
        {
          v70 = v35;
          _libc_free(*v36);
          v35 = v70;
        }
        v71 = v35;
        j_j___libc_free_0((unsigned __int64)v36);
        v35 = v71;
        goto LABEL_97;
      }
      v50 = *(_DWORD *)(v80[0] + 64LL);
      v51 = v50 >> 6;
      if ( v50 >> 6 )
      {
        v52 = *(_QWORD **)v80[0];
        while ( *v52 == -1 )
        {
          if ( ++v52 == (_QWORD *)(*(_QWORD *)v80[0] + 8LL * (v51 - 1) + 8) )
            goto LABEL_100;
        }
      }
      else
      {
LABEL_100:
        v53 = v50 & 0x3F;
        if ( !v53 || *(_QWORD *)(*(_QWORD *)v80[0] + 8LL * v51) == (1LL << v53) - 1 )
          v35 = v74 != v61;
      }
      if ( v80[0] )
      {
        if ( *(_QWORD *)v80[0] != v80[0] + 16LL )
        {
          v68 = v35;
          _libc_free(*(_QWORD *)v80[0]);
          v35 = v68;
        }
        v69 = v35;
        j_j___libc_free_0(v34);
        v35 = v69;
      }
      v36 = (unsigned __int64 *)v79;
      if ( (v79 & 1) == 0 && v79 )
        goto LABEL_94;
LABEL_97:
      if ( v35 )
        goto LABEL_30;
      v37 = v73;
      if ( !v73 )
      {
LABEL_99:
        v73 = v29;
        goto LABEL_52;
      }
LABEL_48:
      if ( v29 == v37 )
        goto LABEL_99;
      if ( v29 != v62 && v62 )
        goto LABEL_57;
      v62 = v29;
      *(_DWORD *)(v26 + *(_QWORD *)a3) += v72;
LABEL_52:
      if ( v76 != 2 )
        break;
LABEL_30:
      ++v27;
      v26 += 4LL;
      if ( v27 == v77 )
        goto LABEL_54;
    }
    v38 = *(_DWORD *)(*(_QWORD *)a3 + v26) % v72 != v27++;
    v26 += 4LL;
    v76 = v38 + 1;
  }
  while ( v27 != v77 );
LABEL_54:
  if ( v76 != 1 )
  {
    if ( v62 )
    {
      v54 = 6;
      goto LABEL_111;
    }
LABEL_110:
    v54 = 7;
LABEL_111:
    LODWORD(v80[0]) = v54;
    BYTE4(v80[0]) = 1;
    return v80[0];
  }
  if ( !v62 )
    goto LABEL_110;
  LODWORD(v80[0]) = 2;
  BYTE4(v80[0]) = 1;
  return v80[0];
}
