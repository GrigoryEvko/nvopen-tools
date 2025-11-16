// Function: sub_A87430
// Address: 0xa87430
//
__int64 __fastcall sub_A87430(_BYTE *a1, __int64 a2)
{
  _BYTE *v2; // rbx
  unsigned __int8 v3; // al
  _BYTE *v4; // r15
  __int64 v5; // r13
  _BYTE **v6; // r12
  __int64 v7; // r13
  _BYTE **v8; // r14
  __int64 v9; // rax
  __int64 v10; // r13
  _BYTE **v11; // r13
  unsigned __int8 v12; // al
  unsigned __int64 v13; // rdx
  __int64 *v14; // r9
  __int64 *v15; // r12
  __int64 *v16; // r14
  __int64 v17; // rbx
  unsigned __int8 v18; // al
  _QWORD *v19; // rdx
  __int64 v20; // r13
  _QWORD *v21; // rax
  _BYTE *v22; // r10
  unsigned __int64 v23; // rdx
  __int64 v24; // r15
  _BYTE *v25; // r15
  unsigned int v26; // eax
  __int64 v27; // rdx
  unsigned __int64 v28; // rdx
  __int64 v29; // rcx
  _QWORD *v30; // r10
  __int64 v31; // rax
  unsigned __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // r10
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  __int64 v37; // rdx
  unsigned __int8 v38; // cl
  bool v39; // si
  int v40; // r11d
  _BYTE *v41; // r8
  __int64 v42; // r14
  __int64 *v43; // r15
  unsigned int v44; // r13d
  int v45; // r12d
  __int64 v46; // rsi
  __int64 v47; // r9
  __int64 result; // rax
  __int64 v49; // rax
  unsigned __int64 v50; // rdx
  __int64 v51; // rdx
  _BYTE *v52; // rsi
  _QWORD *v53; // rdi
  _QWORD *v54; // rdi
  __int64 v55; // r13
  unsigned int v56; // eax
  bool v57; // r8
  __int64 v58; // [rsp+0h] [rbp-150h]
  _BYTE *v59; // [rsp+8h] [rbp-148h]
  _BYTE *v61; // [rsp+20h] [rbp-130h]
  _QWORD *v62; // [rsp+20h] [rbp-130h]
  __int64 v63; // [rsp+20h] [rbp-130h]
  __int64 *v64; // [rsp+20h] [rbp-130h]
  __int64 v65; // [rsp+20h] [rbp-130h]
  __int64 v66; // [rsp+28h] [rbp-128h]
  _QWORD v67[2]; // [rsp+30h] [rbp-120h] BYREF
  __int64 v68; // [rsp+40h] [rbp-110h] BYREF
  _QWORD v69[4]; // [rsp+50h] [rbp-100h] BYREF
  __int16 v70; // [rsp+70h] [rbp-E0h]
  _BYTE *v71; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v72; // [rsp+88h] [rbp-C8h]
  _BYTE v73[64]; // [rsp+90h] [rbp-C0h] BYREF
  _BYTE *v74; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v75; // [rsp+D8h] [rbp-78h]
  _BYTE v76[112]; // [rsp+E0h] [rbp-70h] BYREF

  v2 = a1;
  if ( *a1 != 5 )
    return (__int64)a1;
  v3 = *(a1 - 16);
  v4 = a1 - 16;
  if ( (v3 & 2) != 0 )
  {
    v6 = (_BYTE **)*((_QWORD *)a1 - 4);
    v5 = *((unsigned int *)a1 - 6);
  }
  else
  {
    v5 = (*((_WORD *)a1 - 8) >> 6) & 0xF;
    v6 = (_BYTE **)&v4[-8 * ((v3 >> 2) & 0xF)];
  }
  v7 = 8 * v5;
  v8 = &v6[(unsigned __int64)v7 / 8];
  v9 = v7 >> 3;
  v10 = v7 >> 5;
  if ( v10 )
  {
    v11 = &v6[4 * v10];
    while ( 1 )
    {
      if ( sub_A7C6B0(*v6, a2) )
        goto LABEL_11;
      if ( sub_A7C6B0(v6[1], a2) )
      {
        ++v6;
        goto LABEL_11;
      }
      if ( sub_A7C6B0(v6[2], a2) )
      {
        v6 += 2;
        goto LABEL_11;
      }
      if ( sub_A7C6B0(v6[3], a2) )
        break;
      v6 += 4;
      if ( v11 == v6 )
      {
        v9 = v8 - v6;
        goto LABEL_49;
      }
    }
    v6 += 3;
    goto LABEL_11;
  }
LABEL_49:
  if ( v9 == 2 )
    goto LABEL_86;
  if ( v9 == 3 )
  {
    if ( sub_A7C6B0(*v6, a2) )
      goto LABEL_11;
    ++v6;
LABEL_86:
    if ( sub_A7C6B0(*v6, a2) )
      goto LABEL_11;
    ++v6;
    goto LABEL_88;
  }
  if ( v9 != 1 )
    return (__int64)a1;
LABEL_88:
  v57 = sub_A7C6B0(*v6, a2);
  result = (__int64)a1;
  if ( !v57 )
    return result;
LABEL_11:
  if ( v8 == v6 )
    return (__int64)a1;
  v71 = v73;
  v72 = 0x800000000LL;
  v12 = *(a1 - 16);
  if ( (v12 & 2) == 0 )
  {
    v13 = (*((_WORD *)a1 - 8) >> 6) & 0xF;
    if ( v13 <= 8 )
      goto LABEL_58;
    goto LABEL_56;
  }
  v13 = *((unsigned int *)a1 - 6);
  if ( v13 > 8 )
  {
LABEL_56:
    a2 = (__int64)v73;
    sub_C8D5F0(&v71, v73, v13, 8);
    v12 = *(a1 - 16);
    if ( (v12 & 2) != 0 )
    {
      v13 = *((unsigned int *)a1 - 6);
      goto LABEL_14;
    }
    v13 = (*((_WORD *)a1 - 8) >> 6) & 0xF;
LABEL_58:
    v14 = (__int64 *)&v4[-8 * ((v12 >> 2) & 0xF)];
    goto LABEL_15;
  }
LABEL_14:
  v14 = (__int64 *)*((_QWORD *)a1 - 4);
LABEL_15:
  v15 = &v14[v13];
  if ( v15 != v14 )
  {
    v16 = v14;
    while ( 1 )
    {
      v17 = *v16;
      if ( !*v16 )
      {
        v24 = 0;
        goto LABEL_60;
      }
      if ( *(_BYTE *)v17 != 5 )
        break;
      v18 = *(_BYTE *)(v17 - 16);
      if ( (v18 & 2) != 0 )
      {
        if ( !*(_DWORD *)(v17 - 24) )
          break;
        v19 = *(_QWORD **)(v17 - 32);
        v20 = v17 - 16;
      }
      else
      {
        if ( (*(_WORD *)(v17 - 16) & 0x3C0) == 0 )
          break;
        v20 = v17 - 16;
        v19 = (_QWORD *)(v17 - 16 - 8LL * ((v18 >> 2) & 0xF));
      }
      if ( !*v19 )
        break;
      if ( *(_BYTE *)*v19 )
        break;
      v61 = (_BYTE *)*v19;
      v21 = (_QWORD *)sub_B91420(*v19, a2);
      v22 = v61;
      if ( v23 <= 0xF )
        break;
      v24 = v17;
      a2 = *v21 ^ 0x6365762E6D766C6CLL;
      if ( !(a2 | v21[1] ^ 0x2E72657A69726F74LL) )
      {
        v25 = v76;
        v74 = v76;
        v75 = 0x800000000LL;
        if ( (*(_BYTE *)(v17 - 16) & 2) != 0 )
        {
          v26 = *(_DWORD *)(v17 - 24);
          v27 = v26;
          if ( v26 <= 8 )
            goto LABEL_28;
LABEL_79:
          a2 = (__int64)v76;
          sub_C8D5F0(&v74, v76, v27, 8);
          v22 = v61;
        }
        else
        {
          v56 = (*(_WORD *)(v17 - 16) >> 6) & 0xF;
          v27 = v56;
          if ( v56 > 8 )
            goto LABEL_79;
        }
LABEL_28:
        v29 = sub_B91420(v22, a2);
        v30 = (_QWORD *)(*(_QWORD *)(v17 + 8) & 0xFFFFFFFFFFFFFFF8LL);
        if ( (*(_QWORD *)(v17 + 8) & 4) != 0 )
          v30 = (_QWORD *)*v30;
        v31 = v28;
        if ( v28 == 22 )
        {
          if ( *(_QWORD *)v29 ^ 0x6365762E6D766C6CLL | *(_QWORD *)(v29 + 8) ^ 0x2E72657A69726F74LL
            || *(_DWORD *)(v29 + 16) != 1869770357
            || *(_WORD *)(v29 + 20) != 27756 )
          {
LABEL_32:
            v32 = v28 - 16;
            v31 = 16;
            goto LABEL_33;
          }
          v34 = sub_B9B140(v30, "llvm.loop.interleave.count", 26);
        }
        else
        {
          v32 = 0;
          if ( v28 > 0xF )
            goto LABEL_32;
LABEL_33:
          v69[3] = v32;
          v69[2] = v29 + v31;
          v62 = v30;
          v69[0] = "llvm.loop.vectorize.";
          v70 = 1283;
          sub_CA0F50(v67, v69);
          v33 = sub_B9B140(v62, v67[0], v67[1]);
          v34 = v33;
          if ( (__int64 *)v67[0] != &v68 )
          {
            v63 = v33;
            j_j___libc_free_0(v67[0], v68 + 1);
            v34 = v63;
          }
        }
        v35 = (unsigned int)v75;
        v36 = (unsigned int)v75 + 1LL;
        if ( v36 > HIDWORD(v75) )
        {
          v65 = v34;
          sub_C8D5F0(&v74, v76, v36, 8);
          v35 = (unsigned int)v75;
          v34 = v65;
        }
        *(_QWORD *)&v74[8 * v35] = v34;
        v37 = (unsigned int)(v75 + 1);
        LODWORD(v75) = v75 + 1;
        v38 = *(_BYTE *)(v17 - 16);
        v39 = (v38 & 2) != 0;
        if ( (v38 & 2) != 0 )
          v40 = *(_DWORD *)(v17 - 24);
        else
          v40 = (*(_WORD *)(v17 - 16) >> 6) & 0xF;
        if ( v40 != 1 )
        {
          v64 = v16;
          v41 = v76;
          v42 = v20;
          v43 = v15;
          v44 = 1;
          v45 = v40;
          while ( 1 )
          {
            if ( v39 )
              v46 = *(_QWORD *)(v17 - 32);
            else
              v46 = v42 - 8LL * ((v38 >> 2) & 0xF);
            v47 = *(_QWORD *)(v46 + 8LL * v44);
            if ( v37 + 1 > (unsigned __int64)HIDWORD(v75) )
            {
              v58 = *(_QWORD *)(v46 + 8LL * v44);
              v59 = v41;
              sub_C8D5F0(&v74, v41, v37 + 1, 8);
              v37 = (unsigned int)v75;
              v47 = v58;
              v41 = v59;
            }
            ++v44;
            *(_QWORD *)&v74[8 * v37] = v47;
            v37 = (unsigned int)(v75 + 1);
            LODWORD(v75) = v75 + 1;
            if ( v45 == v44 )
              break;
            v38 = *(_BYTE *)(v17 - 16);
            v39 = (v38 & 2) != 0;
          }
          v16 = v64;
          v15 = v43;
          v25 = v41;
        }
        a2 = (__int64)v74;
        v54 = (_QWORD *)(*(_QWORD *)(v17 + 8) & 0xFFFFFFFFFFFFFFF8LL);
        if ( (*(_QWORD *)(v17 + 8) & 4) != 0 )
          v54 = (_QWORD *)*v54;
        v55 = sub_B9C770(v54, v74, v37, 0, 1);
        if ( v74 != v25 )
          _libc_free(v74, a2);
        v24 = v55;
      }
LABEL_60:
      v49 = (unsigned int)v72;
      v50 = (unsigned int)v72 + 1LL;
      if ( v50 > HIDWORD(v72) )
      {
        a2 = (__int64)v73;
        sub_C8D5F0(&v71, v73, v50, 8);
        v49 = (unsigned int)v72;
      }
      ++v16;
      *(_QWORD *)&v71[8 * v49] = v24;
      v51 = (unsigned int)(v72 + 1);
      LODWORD(v72) = v72 + 1;
      if ( v15 == v16 )
      {
        v2 = a1;
        goto LABEL_64;
      }
    }
    v24 = v17;
    goto LABEL_60;
  }
  v51 = (unsigned int)v72;
LABEL_64:
  v52 = v71;
  v53 = (_QWORD *)(*((_QWORD *)v2 + 1) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*((_QWORD *)v2 + 1) & 4) != 0 )
    v53 = (_QWORD *)*v53;
  result = sub_B9C770(v53, v71, v51, 0, 1);
  if ( v71 != v73 )
  {
    v66 = result;
    _libc_free(v71, v52);
    return v66;
  }
  return result;
}
