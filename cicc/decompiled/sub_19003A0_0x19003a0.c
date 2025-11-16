// Function: sub_19003A0
// Address: 0x19003a0
//
void __fastcall sub_19003A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi
  __int64 v3; // rdx
  unsigned __int64 *v4; // rax
  _QWORD *v5; // rdi
  int v6; // r14d
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // r12
  int v10; // esi
  _QWORD *v11; // rdi
  unsigned int v12; // edx
  _QWORD *v13; // r8
  _QWORD *v14; // r9
  _QWORD *v15; // rax
  unsigned int v16; // esi
  int v17; // r9d
  unsigned __int64 v18; // r12
  _BYTE *v19; // r15
  __int64 v20; // rbx
  __int64 v21; // rdi
  char *v22; // rax
  char *v23; // rsi
  __int64 v24; // rdx
  signed __int64 v25; // rcx
  __int64 v26; // r8
  char *v27; // rcx
  unsigned int v28; // eax
  _QWORD *v29; // rax
  _BYTE **v30; // rdx
  __int64 v31; // rax
  unsigned int v32; // edx
  _QWORD *v33; // r10
  unsigned int v34; // edi
  __int64 v35; // r8
  __int64 v36; // rdx
  int v37; // r11d
  int v38; // esi
  unsigned int v39; // ecx
  int v40; // edi
  _QWORD *v41; // rdx
  int v42; // esi
  unsigned int v43; // ecx
  int v44; // edi
  int v45; // r11d
  __int64 v46; // rdi
  unsigned int v47; // eax
  unsigned int v48; // ebx
  char v49; // al
  __int64 v50; // rdi
  _BYTE **v51; // rdx
  _QWORD *v52; // rax
  __int64 v53; // rax
  _QWORD *v54; // rax
  _BYTE **v55; // rdx
  int v56; // r11d
  __int64 v57; // rdi
  _QWORD *v58; // [rsp+8h] [rbp-118h]
  _QWORD *v59; // [rsp+8h] [rbp-118h]
  _QWORD *v60; // [rsp+8h] [rbp-118h]
  _QWORD *v62; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v63; // [rsp+48h] [rbp-D8h]
  _QWORD v64[8]; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v65; // [rsp+90h] [rbp-90h] BYREF
  __int64 v66; // [rsp+98h] [rbp-88h]
  _QWORD *v67; // [rsp+A0h] [rbp-80h] BYREF
  unsigned int v68; // [rsp+A8h] [rbp-78h]
  _BYTE *v69; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v70; // [rsp+C8h] [rbp-58h]
  _BYTE v71[80]; // [rsp+D0h] [rbp-50h] BYREF

  v2 = *(_QWORD *)(a1 + 72);
  if ( !v2 )
    return;
  v3 = sub_1422850(v2, a2);
  if ( !v3 )
    return;
  v4 = (unsigned __int64 *)&v67;
  v65 = 0;
  v66 = 1;
  do
    *v4++ = -8;
  while ( v4 != (unsigned __int64 *)&v69 );
  v5 = v64;
  v6 = 0;
  v64[0] = v3;
  v69 = v71;
  v62 = v64;
  v70 = 0x400000000LL;
  v63 = 0x800000001LL;
  v7 = 0;
  do
  {
    v8 = v5[v7];
    v9 = *(_QWORD *)(v8 + 8);
    if ( v9 )
    {
      while ( 1 )
      {
        v15 = sub_1648700(v9);
        if ( *((_BYTE *)v15 + 16) == 23 )
        {
          if ( (v66 & 1) != 0 )
          {
            v10 = 3;
            v11 = &v67;
          }
          else
          {
            v16 = v68;
            v11 = v67;
            if ( !v68 )
            {
              v32 = v66;
              ++v65;
              v33 = 0;
              v34 = ((unsigned int)v66 >> 1) + 1;
              goto LABEL_54;
            }
            v10 = v68 - 1;
          }
          v12 = v10 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
          v13 = &v11[v12];
          v14 = (_QWORD *)*v13;
          if ( v15 != (_QWORD *)*v13 )
            break;
        }
LABEL_10:
        v9 = *(_QWORD *)(v9 + 8);
        if ( !v9 )
          goto LABEL_15;
      }
      v37 = 1;
      v33 = 0;
      while ( v14 != (_QWORD *)-8LL )
      {
        if ( v33 || v14 != (_QWORD *)-16LL )
          v13 = v33;
        v12 = v10 & (v37 + v12);
        v14 = (_QWORD *)v11[v12];
        if ( v15 == v14 )
          goto LABEL_10;
        ++v37;
        v33 = v13;
        v13 = &v11[v12];
      }
      v32 = v66;
      if ( !v33 )
        v33 = v13;
      ++v65;
      v34 = ((unsigned int)v66 >> 1) + 1;
      if ( (v66 & 1) != 0 )
      {
        LODWORD(v35) = 4 * v34;
        v16 = 4;
        if ( 4 * v34 >= 0xC )
        {
LABEL_71:
          v58 = v15;
          sub_18FFFF0((__int64)&v65, 2 * v16);
          v15 = v58;
          if ( (v66 & 1) != 0 )
          {
            v38 = 3;
            v14 = &v67;
          }
          else
          {
            v14 = v67;
            if ( !v68 )
            {
LABEL_132:
              LODWORD(v66) = (2 * ((unsigned int)v66 >> 1) + 2) | v66 & 1;
              BUG();
            }
            v38 = v68 - 1;
          }
          v39 = v38 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
          v33 = &v14[v39];
          v32 = v66;
          v35 = *v33;
          if ( v58 == (_QWORD *)*v33 )
            goto LABEL_56;
          v40 = 1;
          v41 = 0;
          while ( v35 != -8 )
          {
            if ( v35 == -16 && !v41 )
              v41 = v33;
            v56 = v40 + 1;
            v57 = v38 & (v39 + v40);
            v33 = &v14[v57];
            v39 = v57;
            v35 = *v33;
            if ( v58 == (_QWORD *)*v33 )
              goto LABEL_78;
            v40 = v56;
          }
          goto LABEL_76;
        }
      }
      else
      {
        v16 = v68;
LABEL_54:
        LODWORD(v35) = 4 * v34;
        if ( 4 * v34 >= 3 * v16 )
          goto LABEL_71;
      }
      if ( v16 - HIDWORD(v66) - v34 > v16 >> 3 )
        goto LABEL_56;
      v60 = v15;
      sub_18FFFF0((__int64)&v65, v16);
      v15 = v60;
      if ( (v66 & 1) != 0 )
      {
        v42 = 3;
        v14 = &v67;
      }
      else
      {
        v14 = v67;
        if ( !v68 )
          goto LABEL_132;
        v42 = v68 - 1;
      }
      v43 = v42 & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
      v33 = &v14[v43];
      v32 = v66;
      v35 = *v33;
      if ( v60 == (_QWORD *)*v33 )
        goto LABEL_56;
      v44 = 1;
      v41 = 0;
      while ( v35 != -8 )
      {
        if ( v35 == -16 && !v41 )
          v41 = v33;
        v45 = v44 + 1;
        v46 = v42 & (v43 + v44);
        v33 = &v14[v46];
        v43 = v46;
        v35 = *v33;
        if ( v60 == (_QWORD *)*v33 )
          goto LABEL_78;
        v44 = v45;
      }
LABEL_76:
      if ( v41 )
        v33 = v41;
LABEL_78:
      v32 = v66;
LABEL_56:
      LODWORD(v66) = (2 * (v32 >> 1) + 2) | v32 & 1;
      if ( *v33 != -8 )
        --HIDWORD(v66);
      *v33 = v15;
      v36 = (unsigned int)v70;
      if ( (unsigned int)v70 >= HIDWORD(v70) )
      {
        v59 = v15;
        sub_16CD150((__int64)&v69, v71, 0, 8, v35, (int)v14);
        v36 = (unsigned int)v70;
        v15 = v59;
      }
      *(_QWORD *)&v69[8 * v36] = v15;
      LODWORD(v70) = v70 + 1;
      goto LABEL_10;
    }
LABEL_15:
    sub_386B550(*(_QWORD *)(a1 + 80), v8);
    v18 = (unsigned __int64)v69;
    v19 = &v69[8 * (unsigned int)v70];
    if ( v19 != v69 )
    {
      do
      {
        v20 = *(_QWORD *)v18;
        v21 = 24LL * (*(_DWORD *)(*(_QWORD *)v18 + 20LL) & 0xFFFFFFF);
        if ( (*(_BYTE *)(*(_QWORD *)v18 + 23LL) & 0x40) != 0 )
        {
          v22 = *(char **)(v20 - 8);
          v23 = &v22[v21];
        }
        else
        {
          v23 = *(char **)v18;
          v22 = (char *)(v20 - v21);
        }
        v24 = *(_QWORD *)v22;
        v25 = 0xAAAAAAAAAAAAAAABLL * (v21 >> 3);
        v26 = v25 >> 2;
        if ( v25 >> 2 )
        {
          v27 = &v22[96 * v26];
          while ( 1 )
          {
            if ( v24 != *((_QWORD *)v22 + 3) )
            {
              v22 += 24;
              goto LABEL_25;
            }
            if ( v24 != *((_QWORD *)v22 + 6) )
            {
              v22 += 48;
              goto LABEL_25;
            }
            if ( v24 != *((_QWORD *)v22 + 9) )
            {
              v22 += 72;
              goto LABEL_25;
            }
            v22 += 96;
            if ( v27 == v22 )
              break;
            if ( v24 != *(_QWORD *)v22 )
              goto LABEL_25;
          }
          v25 = 0xAAAAAAAAAAAAAAABLL * ((v23 - v22) >> 3);
          if ( v23 - v22 == 48 )
            goto LABEL_63;
        }
        else if ( v21 == 48 )
        {
          goto LABEL_64;
        }
        if ( v25 == 3 )
        {
          if ( v24 != *(_QWORD *)v22 )
            goto LABEL_25;
          v22 += 24;
LABEL_63:
          if ( v24 != *(_QWORD *)v22 )
            goto LABEL_25;
LABEL_64:
          v22 += 24;
          goto LABEL_46;
        }
        if ( v25 != 1 )
          goto LABEL_47;
LABEL_46:
        if ( v24 == *(_QWORD *)v22 )
        {
LABEL_47:
          v31 = (unsigned int)v63;
          if ( (unsigned int)v63 >= HIDWORD(v63) )
          {
            sub_16CD150((__int64)&v62, v64, 0, 8, v26, v17);
            v31 = (unsigned int)v63;
          }
          v62[v31] = v20;
          LODWORD(v63) = v63 + 1;
          goto LABEL_26;
        }
LABEL_25:
        if ( v22 == v23 )
          goto LABEL_47;
LABEL_26:
        v18 += 8LL;
      }
      while ( v19 != (_BYTE *)v18 );
    }
    ++v65;
    v28 = (unsigned int)v66 >> 1;
    if ( !((unsigned int)v66 >> 1) && !HIDWORD(v66) )
      goto LABEL_35;
    if ( (v66 & 1) != 0 )
    {
      v30 = &v69;
      v29 = &v67;
      goto LABEL_33;
    }
    if ( 4 * v28 >= v68 || v68 <= 0x40 )
    {
      v29 = v67;
      v30 = (_BYTE **)&v67[v68];
      if ( v67 == v30 )
      {
LABEL_34:
        v66 &= 1u;
        goto LABEL_35;
      }
      do
LABEL_33:
        *v29++ = -8;
      while ( v29 != v30 );
      goto LABEL_34;
    }
    if ( v28 && (v47 = v28 - 1) != 0 )
    {
      _BitScanReverse(&v47, v47);
      v48 = 1 << (33 - (v47 ^ 0x1F));
      if ( v48 - 5 > 0x3A )
      {
        if ( v68 == v48 )
        {
          v66 &= 1u;
          if ( v66 )
          {
            v55 = &v69;
            v54 = &v67;
          }
          else
          {
            v54 = v67;
            v55 = (_BYTE **)&v67[v68];
          }
          do
          {
            if ( v54 )
              *v54 = -8;
            ++v54;
          }
          while ( v54 != v55 );
          goto LABEL_35;
        }
        j___libc_free_0(v67);
        v49 = v66 | 1;
        LOBYTE(v66) = v66 | 1;
        if ( v48 <= 4 )
          goto LABEL_106;
        v50 = 8LL * v48;
      }
      else
      {
        v48 = 64;
        j___libc_free_0(v67);
        v49 = v66;
        v50 = 512;
      }
      LOBYTE(v66) = v49 & 0xFE;
      v53 = sub_22077B0(v50);
      v68 = v48;
      v67 = (_QWORD *)v53;
    }
    else
    {
      j___libc_free_0(v67);
      LOBYTE(v66) = v66 | 1;
    }
LABEL_106:
    v66 &= 1u;
    if ( v66 )
    {
      v51 = &v69;
      v52 = &v67;
    }
    else
    {
      v52 = v67;
      v51 = (_BYTE **)&v67[v68];
      if ( v67 == v51 )
        goto LABEL_35;
    }
    do
    {
      if ( v52 )
        *v52 = -8;
      ++v52;
    }
    while ( v52 != v51 );
LABEL_35:
    v7 = (unsigned int)(v6 + 1);
    LODWORD(v70) = 0;
    v5 = v62;
    v6 = v7;
  }
  while ( (unsigned int)v63 > (unsigned int)v7 );
  if ( v62 != v64 )
    _libc_free((unsigned __int64)v62);
  if ( v69 != v71 )
    _libc_free((unsigned __int64)v69);
  if ( (v66 & 1) == 0 )
    j___libc_free_0(v67);
}
