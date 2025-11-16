// Function: sub_A45E40
// Address: 0xa45e40
//
__int64 __fastcall sub_A45E40(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 result; // rax
  __int64 v6; // r13
  unsigned __int8 v7; // al
  __int64 v8; // r14
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 *v11; // rax
  __int64 v12; // rdx
  _BYTE *v13; // rcx
  _BYTE *v14; // rdi
  __int64 v15; // r13
  unsigned __int8 v16; // al
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 *v19; // r15
  __int64 v20; // rax
  __int64 v21; // r14
  __int64 *v22; // r14
  unsigned __int8 v23; // al
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // r14
  _BYTE *v27; // rdx
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  unsigned __int8 v30; // al
  __int64 v31; // r13
  __int64 v32; // rax
  _QWORD *v33; // rdx
  __int64 v34; // r14
  int v35; // r11d
  __int64 v36; // r8
  __int64 *v37; // rdx
  unsigned int v38; // ecx
  __int64 *v39; // rax
  __int64 v40; // rdi
  _QWORD *v41; // rax
  __int64 *v42; // r13
  __int64 *v43; // r15
  __int64 v44; // rdx
  __int64 v45; // r8
  _QWORD *v46; // rax
  __int64 v47; // r14
  unsigned __int8 v48; // dl
  __int64 v49; // r8
  int v50; // eax
  int v51; // ecx
  int v52; // r9d
  int v53; // r9d
  __int64 v54; // r10
  __int64 v55; // rax
  __int64 v56; // r8
  int v57; // edi
  int v58; // edi
  int v59; // edi
  __int64 v60; // r8
  __int64 *v61; // r9
  __int64 v62; // r15
  int v63; // eax
  char *v64; // [rsp+18h] [rbp-368h]
  __int64 v65; // [rsp+18h] [rbp-368h]
  __int64 v66; // [rsp+28h] [rbp-358h] BYREF
  __int64 *v67; // [rsp+30h] [rbp-350h] BYREF
  __int64 v68; // [rsp+38h] [rbp-348h]
  _BYTE v69[256]; // [rsp+40h] [rbp-340h] BYREF
  _BYTE *v70; // [rsp+140h] [rbp-240h] BYREF
  __int64 v71; // [rsp+148h] [rbp-238h]
  _BYTE v72[560]; // [rsp+150h] [rbp-230h] BYREF

  v3 = a2;
  v67 = (__int64 *)v69;
  v68 = 0x2000000000LL;
  v70 = v72;
  v71 = 0x2000000000LL;
  result = sub_A45AE0(a1, a2, a3);
  if ( result )
  {
    v6 = result;
    v7 = *(_BYTE *)(result - 16);
    if ( (v7 & 2) != 0 )
      v8 = *(_QWORD *)(v6 - 32);
    else
      v8 = v6 - 8LL * ((v7 >> 2) & 0xF) - 16;
    v9 = (unsigned int)v71;
    v10 = (unsigned int)v71 + 1LL;
    if ( v10 > HIDWORD(v71) )
    {
      a2 = (__int64)v72;
      sub_C8D5F0(&v70, v72, v10, 16);
      v9 = (unsigned int)v71;
    }
    v11 = (__int64 *)&v70[16 * v9];
    *v11 = v6;
    v11[1] = v8;
    result = (unsigned int)v71;
    v12 = (unsigned int)(v71 + 1);
    LODWORD(v71) = v71 + 1;
  }
  else
  {
    v12 = (unsigned int)v71;
  }
  v13 = v70;
  v14 = v70;
  if ( !(_DWORD)v12 )
    goto LABEL_35;
  while ( 2 )
  {
    v15 = *(_QWORD *)&v13[16 * v12 - 16];
    while ( 1 )
    {
      v16 = *(_BYTE *)(v15 - 16);
      a2 = (__int64)&v13[16 * v12 - 16];
      if ( (v16 & 2) != 0 )
      {
        v17 = *(_QWORD *)(v15 - 32);
        v18 = *(unsigned int *)(v15 - 24);
      }
      else
      {
        v18 = (*(_WORD *)(v15 - 16) >> 6) & 0xF;
        v17 = v15 + -16 - 8LL * ((v16 >> 2) & 0xF);
      }
      v19 = *(__int64 **)(a2 + 8);
      v64 = (char *)(v17 + 8 * v18);
      v20 = (v64 - (char *)v19) >> 3;
      v21 = (v64 - (char *)v19) >> 5;
      if ( v21 > 0 )
      {
        v22 = &v19[4 * v21];
        while ( 1 )
        {
          a2 = v3;
          if ( sub_A45AE0(a1, v3, *v19) )
            break;
          a2 = v3;
          if ( sub_A45AE0(a1, v3, v19[1]) )
          {
            v64 = (char *)(v19 + 1);
            goto LABEL_19;
          }
          a2 = v3;
          if ( sub_A45AE0(a1, v3, v19[2]) )
          {
            v64 = (char *)(v19 + 2);
            goto LABEL_19;
          }
          a2 = v3;
          if ( sub_A45AE0(a1, v3, v19[3]) )
          {
            v64 = (char *)(v19 + 3);
            goto LABEL_19;
          }
          v19 += 4;
          if ( v22 == v19 )
          {
            v20 = (v64 - (char *)v19) >> 3;
            goto LABEL_56;
          }
        }
LABEL_18:
        v64 = (char *)v19;
        goto LABEL_19;
      }
LABEL_56:
      if ( v20 != 2 )
      {
        if ( v20 != 3 )
        {
          if ( v20 != 1 )
            goto LABEL_19;
          goto LABEL_59;
        }
        a2 = v3;
        if ( sub_A45AE0(a1, v3, *v19) )
          goto LABEL_18;
        ++v19;
      }
      a2 = v3;
      if ( sub_A45AE0(a1, v3, *v19) )
        goto LABEL_18;
      ++v19;
LABEL_59:
      a2 = v3;
      if ( !sub_A45AE0(a1, v3, *v19) )
        v19 = (__int64 *)v64;
      v64 = (char *)v19;
LABEL_19:
      v23 = *(_BYTE *)(v15 - 16);
      if ( (v23 & 2) != 0 )
      {
        v24 = *(_QWORD *)(v15 - 32);
        v25 = *(unsigned int *)(v15 - 24);
      }
      else
      {
        v25 = (*(_WORD *)(v15 - 16) >> 6) & 0xF;
        v24 = v15 + -16 - 8LL * ((v23 >> 2) & 0xF);
      }
      if ( v64 != (char *)(v24 + 8 * v25) )
        break;
      v66 = v15;
      LODWORD(v71) = v71 - 1;
      sub_A3DCA0(a1 + 208, &v66);
      a2 = *(unsigned int *)(a1 + 280);
      v34 = (__int64)(*(_QWORD *)(a1 + 216) - *(_QWORD *)(a1 + 208)) >> 3;
      if ( !(_DWORD)a2 )
      {
        ++*(_QWORD *)(a1 + 256);
        goto LABEL_85;
      }
      v35 = 1;
      v36 = *(_QWORD *)(a1 + 264);
      v37 = 0;
      v38 = (a2 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v39 = (__int64 *)(v36 + 16LL * v38);
      v40 = *v39;
      if ( *v39 != v15 )
      {
        while ( v40 != -4096 )
        {
          if ( !v37 && v40 == -8192 )
            v37 = v39;
          v38 = (a2 - 1) & (v35 + v38);
          v39 = (__int64 *)(v36 + 16LL * v38);
          v40 = *v39;
          if ( *v39 == v15 )
            goto LABEL_44;
          ++v35;
        }
        if ( !v37 )
          v37 = v39;
        v50 = *(_DWORD *)(a1 + 272);
        ++*(_QWORD *)(a1 + 256);
        v51 = v50 + 1;
        if ( 4 * (v50 + 1) < (unsigned int)(3 * a2) )
        {
          if ( (int)a2 - *(_DWORD *)(a1 + 276) - v51 <= (unsigned int)a2 >> 3 )
          {
            sub_A42F50(a1 + 256, a2);
            v58 = *(_DWORD *)(a1 + 280);
            if ( !v58 )
            {
LABEL_108:
              ++*(_DWORD *)(a1 + 272);
              BUG();
            }
            v59 = v58 - 1;
            v60 = *(_QWORD *)(a1 + 264);
            v61 = 0;
            LODWORD(v62) = v59 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
            v51 = *(_DWORD *)(a1 + 272) + 1;
            v63 = 1;
            v37 = (__int64 *)(v60 + 16LL * (unsigned int)v62);
            a2 = *v37;
            if ( *v37 != v15 )
            {
              while ( a2 != -4096 )
              {
                if ( !v61 && a2 == -8192 )
                  v61 = v37;
                v62 = v59 & (unsigned int)(v62 + v63);
                v37 = (__int64 *)(v60 + 16 * v62);
                a2 = *v37;
                if ( *v37 == v15 )
                  goto LABEL_81;
                ++v63;
              }
              if ( v61 )
                v37 = v61;
            }
          }
          goto LABEL_81;
        }
LABEL_85:
        a2 = (unsigned int)(2 * a2);
        sub_A42F50(a1 + 256, a2);
        v52 = *(_DWORD *)(a1 + 280);
        if ( !v52 )
          goto LABEL_108;
        v53 = v52 - 1;
        v54 = *(_QWORD *)(a1 + 264);
        v51 = *(_DWORD *)(a1 + 272) + 1;
        LODWORD(v55) = v53 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v37 = (__int64 *)(v54 + 16LL * (unsigned int)v55);
        v56 = *v37;
        if ( *v37 != v15 )
        {
          v57 = 1;
          a2 = 0;
          while ( v56 != -4096 )
          {
            if ( v56 == -8192 && !a2 )
              a2 = (__int64)v37;
            v55 = v53 & (unsigned int)(v55 + v57);
            v37 = (__int64 *)(v54 + 16 * v55);
            v56 = *v37;
            if ( *v37 == v15 )
              goto LABEL_81;
            ++v57;
          }
          if ( a2 )
            v37 = (__int64 *)a2;
        }
LABEL_81:
        *(_DWORD *)(a1 + 272) = v51;
        if ( *v37 != -4096 )
          --*(_DWORD *)(a1 + 276);
        *v37 = v15;
        v41 = v37 + 1;
        v37[1] = 0;
        goto LABEL_45;
      }
LABEL_44:
      v41 = v39 + 1;
LABEL_45:
      *((_DWORD *)v41 + 1) = v34;
      result = (unsigned int)v71;
      if ( (_DWORD)v71 )
      {
        v12 = (unsigned int)v71;
        v13 = v70;
        v15 = *(_QWORD *)&v70[16 * (unsigned int)v71 - 16];
        a2 = *(_BYTE *)(v15 + 1) & 0x7F;
        if ( (_BYTE)a2 != 1 )
          continue;
      }
      v42 = v67;
      v43 = &v67[(unsigned int)v68];
      if ( v43 == v67 )
      {
LABEL_65:
        LODWORD(v68) = 0;
        goto LABEL_27;
      }
      while ( 1 )
      {
        v47 = *v42;
        v48 = *(_BYTE *)(*v42 - 16);
        if ( (v48 & 2) != 0 )
        {
          v44 = result + 1;
          v45 = *(_QWORD *)(v47 - 32);
          if ( result + 1 > (unsigned __int64)HIDWORD(v71) )
            goto LABEL_53;
        }
        else
        {
          v49 = -16 - 8LL * ((v48 >> 2) & 0xF);
          v44 = result + 1;
          v45 = v47 + v49;
          if ( result + 1 > (unsigned __int64)HIDWORD(v71) )
          {
LABEL_53:
            a2 = (__int64)v72;
            v65 = v45;
            sub_C8D5F0(&v70, v72, v44, 16);
            result = (unsigned int)v71;
            v45 = v65;
          }
        }
        v46 = &v70[16 * result];
        ++v42;
        *v46 = v47;
        v46[1] = v45;
        result = (unsigned int)(v71 + 1);
        LODWORD(v71) = v71 + 1;
        if ( v43 == v42 )
          goto LABEL_65;
      }
    }
    v26 = *(_QWORD *)v64;
    v27 = v70;
    *(_QWORD *)&v70[16 * (unsigned int)v71 - 8] = v64 + 8;
    if ( (*(_BYTE *)(v26 + 1) & 0x7F) != 1 || (*(_BYTE *)(v15 + 1) & 0x7F) == 1 )
    {
      v30 = *(_BYTE *)(v26 - 16);
      if ( (v30 & 2) != 0 )
        v31 = *(_QWORD *)(v26 - 32);
      else
        v31 = v26 + -16 - 8LL * ((v30 >> 2) & 0xF);
      v32 = (unsigned int)v71;
      if ( (unsigned __int64)(unsigned int)v71 + 1 > HIDWORD(v71) )
      {
        a2 = (__int64)v72;
        sub_C8D5F0(&v70, v72, (unsigned int)v71 + 1LL, 16);
        v27 = v70;
        v32 = (unsigned int)v71;
      }
      v33 = &v27[16 * v32];
      *v33 = v26;
      v13 = v70;
      v33[1] = v31;
      v14 = v13;
      result = (unsigned int)(v71 + 1);
      LODWORD(v71) = result;
      if ( (_DWORD)result )
        goto LABEL_28;
    }
    else
    {
      v28 = (unsigned int)v68;
      v29 = (unsigned int)v68 + 1LL;
      if ( v29 > HIDWORD(v68) )
      {
        a2 = (__int64)v69;
        sub_C8D5F0(&v67, v69, v29, 8);
        v28 = (unsigned int)v68;
      }
      v67[v28] = v26;
      result = (unsigned int)v71;
      LODWORD(v68) = v68 + 1;
LABEL_27:
      v13 = v70;
      v14 = v70;
      if ( (_DWORD)result )
      {
LABEL_28:
        v12 = (unsigned int)result;
        continue;
      }
    }
    break;
  }
LABEL_35:
  if ( v14 != v72 )
    result = _libc_free(v14, a2);
  if ( v67 != (__int64 *)v69 )
    return _libc_free(v67, a2);
  return result;
}
