// Function: sub_335ED90
// Address: 0x335ed90
//
void __fastcall sub_335ED90(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  int v10; // edx
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rcx
  __int64 v14; // rax
  __int64 *v15; // rdx
  unsigned int v16; // eax
  __int64 v17; // rcx
  __int64 v18; // rbx
  __int64 *v19; // r14
  __int64 *v20; // r15
  __int64 v21; // rsi
  __int64 *v22; // rax
  unsigned int v23; // ecx
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rdx
  __int64 v31; // r14
  int v32; // eax
  unsigned int *v33; // rax
  int v34; // eax
  __int64 v35; // r15
  __int64 v36; // rax
  __int64 v37; // rdx
  int v38; // eax
  int v39; // eax
  _BYTE *v40; // r8
  int v41; // esi
  _BYTE *v42; // rdi
  __int64 *v43; // rax
  __int64 v44; // rax
  __int64 v45; // r10
  unsigned int v46; // ecx
  unsigned __int64 v47; // rdx
  __int64 v48; // rdx
  int v49; // edx
  unsigned int *v50; // rdx
  unsigned __int64 v51; // r14
  __int64 v52; // r13
  __int64 v53; // rbx
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rax
  unsigned __int64 v57; // rdx
  unsigned __int64 v58; // rbx
  unsigned __int64 v59; // rdi
  unsigned __int64 v60; // rdi
  __int64 i; // [rsp+10h] [rbp-3E0h]
  unsigned __int64 v62; // [rsp+20h] [rbp-3D0h]
  __int64 v63; // [rsp+28h] [rbp-3C8h]
  __int64 v64; // [rsp+28h] [rbp-3C8h]
  unsigned __int64 v65; // [rsp+28h] [rbp-3C8h]
  __int64 v66; // [rsp+30h] [rbp-3C0h] BYREF
  int v67; // [rsp+38h] [rbp-3B8h]
  _BYTE *v68; // [rsp+40h] [rbp-3B0h] BYREF
  __int64 v69; // [rsp+48h] [rbp-3A8h]
  _BYTE v70[64]; // [rsp+50h] [rbp-3A0h] BYREF
  __int64 v71; // [rsp+90h] [rbp-360h] BYREF
  __int64 *v72; // [rsp+98h] [rbp-358h]
  __int64 v73; // [rsp+A0h] [rbp-350h]
  int v74; // [rsp+A8h] [rbp-348h]
  char v75; // [rsp+ACh] [rbp-344h]
  __int64 v76; // [rsp+B0h] [rbp-340h] BYREF
  __int64 *v77; // [rsp+1B0h] [rbp-240h] BYREF
  __int64 v78; // [rsp+1B8h] [rbp-238h]
  _QWORD v79[70]; // [rsp+1C0h] [rbp-230h] BYREF

  v7 = a1[74];
  v8 = *(_QWORD *)(v7 + 408);
  v9 = v7 + 400;
  if ( v8 == v7 + 400 )
    goto LABEL_7;
  v10 = 0;
  do
  {
    if ( !v8 )
    {
      MEMORY[0x24] = 0;
      BUG();
    }
    *(_DWORD *)(v8 + 28) = -1;
    v8 = *(_QWORD *)(v8 + 8);
    ++v10;
  }
  while ( v8 != v9 );
  v11 = a1[6];
  v12 = (unsigned int)(2 * v10);
  v13 = (__int64)(a1[8] - v11) >> 8;
  if ( v12 > v13 )
  {
    v51 = a1[7];
    v62 = v12 << 8;
    v65 = v51 - v11;
    if ( 2 * v10 )
    {
      v52 = sub_22077B0(v12 << 8);
      if ( v11 == v51 )
      {
LABEL_88:
        v58 = a1[7];
        v51 = a1[6];
        if ( v58 != v51 )
        {
          do
          {
            v59 = *(_QWORD *)(v51 + 120);
            if ( v59 != v51 + 136 )
              _libc_free(v59);
            v60 = *(_QWORD *)(v51 + 40);
            if ( v60 != v51 + 56 )
              _libc_free(v60);
            v51 += 256LL;
          }
          while ( v58 != v51 );
          v51 = a1[6];
        }
        goto LABEL_85;
      }
    }
    else
    {
      v52 = 0;
      if ( v11 == v51 )
      {
LABEL_85:
        if ( v51 )
          j_j___libc_free_0(v51);
        a1[6] = v52;
        a1[7] = v52 + v65;
        a1[8] = v62 + v52;
        goto LABEL_6;
      }
    }
    v53 = v52;
    do
    {
      if ( v53 )
      {
        *(_QWORD *)v53 = *(_QWORD *)v11;
        *(_QWORD *)(v53 + 8) = *(_QWORD *)(v11 + 8);
        *(_QWORD *)(v53 + 16) = *(_QWORD *)(v11 + 16);
        *(_QWORD *)(v53 + 24) = *(_QWORD *)(v11 + 24);
        v54 = *(_QWORD *)(v11 + 32);
        *(_DWORD *)(v53 + 48) = 0;
        *(_QWORD *)(v53 + 32) = v54;
        *(_QWORD *)(v53 + 40) = v53 + 56;
        *(_DWORD *)(v53 + 52) = 4;
        v55 = *(unsigned int *)(v11 + 48);
        if ( (_DWORD)v55 )
          sub_335BB20(v53 + 40, v11 + 40, v55, v13, a5, a6);
        *(_DWORD *)(v53 + 128) = 0;
        *(_QWORD *)(v53 + 120) = v53 + 136;
        *(_DWORD *)(v53 + 132) = 4;
        if ( *(_DWORD *)(v11 + 128) )
          sub_335BB20(v53 + 120, v11 + 120, v55, v13, a5, a6);
        *(_DWORD *)(v53 + 200) = *(_DWORD *)(v11 + 200);
        *(_DWORD *)(v53 + 204) = *(_DWORD *)(v11 + 204);
        *(_DWORD *)(v53 + 208) = *(_DWORD *)(v11 + 208);
        *(_DWORD *)(v53 + 212) = *(_DWORD *)(v11 + 212);
        *(_DWORD *)(v53 + 216) = *(_DWORD *)(v11 + 216);
        *(_DWORD *)(v53 + 220) = *(_DWORD *)(v11 + 220);
        *(_DWORD *)(v53 + 224) = *(_DWORD *)(v11 + 224);
        *(_DWORD *)(v53 + 228) = *(_DWORD *)(v11 + 228);
        *(_DWORD *)(v53 + 232) = *(_DWORD *)(v11 + 232);
        *(_DWORD *)(v53 + 236) = *(_DWORD *)(v11 + 236);
        *(_DWORD *)(v53 + 240) = *(_DWORD *)(v11 + 240);
        *(_DWORD *)(v53 + 244) = *(_DWORD *)(v11 + 244);
        *(_WORD *)(v53 + 248) = *(_WORD *)(v11 + 248);
        *(_WORD *)(v53 + 250) = *(_WORD *)(v11 + 250);
        *(_WORD *)(v53 + 252) = *(_WORD *)(v11 + 252);
        *(_BYTE *)(v53 + 254) = *(_BYTE *)(v11 + 254);
      }
      v11 += 256LL;
      v53 += 256;
    }
    while ( v51 != v11 );
    goto LABEL_88;
  }
LABEL_6:
  v7 = a1[74];
LABEL_7:
  v74 = 0;
  v72 = &v76;
  v14 = *(_QWORD *)(v7 + 384);
  v15 = v79;
  v77 = v79;
  v78 = 0x4000000001LL;
  v79[0] = v14;
  v76 = v14;
  v73 = 0x100000020LL;
  v75 = 1;
  v71 = 1;
  v68 = v70;
  v69 = 0x800000000LL;
  v16 = 1;
  while ( 1 )
  {
    v17 = v16;
    v18 = v15[v16 - 1];
    LODWORD(v78) = v16 - 1;
    v19 = *(__int64 **)(v18 + 40);
    v20 = &v19[5 * *(unsigned int *)(v18 + 64)];
    if ( v20 != v19 )
    {
      while ( 1 )
      {
        v21 = *v19;
        if ( v75 )
        {
          v22 = v72;
          v17 = HIDWORD(v73);
          v15 = &v72[HIDWORD(v73)];
          if ( v72 != v15 )
          {
            while ( v21 != *v22 )
            {
              if ( v15 == ++v22 )
                goto LABEL_26;
            }
            goto LABEL_14;
          }
LABEL_26:
          if ( HIDWORD(v73) < (unsigned int)v73 )
          {
            ++HIDWORD(v73);
            *v15 = v21;
            ++v71;
            goto LABEL_22;
          }
        }
        sub_C8CC70((__int64)&v71, v21, (__int64)v15, v17, a5, a6);
        if ( (_BYTE)v15 )
        {
LABEL_22:
          v25 = (unsigned int)v78;
          v17 = HIDWORD(v78);
          a5 = *v19;
          v26 = (unsigned int)v78 + 1LL;
          if ( v26 > HIDWORD(v78) )
          {
            v64 = *v19;
            sub_C8D5F0((__int64)&v77, v79, v26, 8u, a5, a6);
            v25 = (unsigned int)v78;
            a5 = v64;
          }
          v15 = v77;
          v19 += 5;
          v77[v25] = a5;
          LODWORD(v78) = v78 + 1;
          if ( v20 == v19 )
            break;
        }
        else
        {
LABEL_14:
          v19 += 5;
          if ( v20 == v19 )
            break;
        }
      }
    }
    v23 = *(_DWORD *)(v18 + 24);
    v24 = (0x3FF8000FFE42uLL >> v23) & 1;
    if ( v23 >= 0x2E )
      LOBYTE(v24) = 0;
    if ( v23 != 324 && !(_BYTE)v24 && *(_DWORD *)(v18 + 36) == -1 )
      break;
    v16 = v78;
    if ( !(_DWORD)v78 )
      goto LABEL_47;
LABEL_20:
    v15 = v77;
  }
  v27 = sub_335EAA0(a1, v18);
  v30 = v18;
  v31 = v27;
  v32 = *(_DWORD *)(v18 + 64);
  if ( v32 )
  {
    while ( 1 )
    {
      v33 = (unsigned int *)(*(_QWORD *)(v30 + 40) + 40LL * (unsigned int)(v32 - 1));
      v30 = *(_QWORD *)v33;
      if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v33 + 48LL) + 16LL * v33[2]) != 262 )
        break;
      *(_DWORD *)(v30 + 36) = *(_DWORD *)(v31 + 200);
      v34 = *(_DWORD *)(v30 + 24);
      if ( v34 >= 0 || *(char *)(*(_QWORD *)(a1[2] + 8) - 40LL * (unsigned int)~v34 + 24) >= 0 )
      {
        v32 = *(_DWORD *)(v30 + 64);
        if ( !v32 )
          break;
      }
      else
      {
        *(_BYTE *)(v31 + 248) |= 2u;
        v32 = *(_DWORD *)(v30 + 64);
        if ( !v32 )
          break;
      }
    }
  }
  v35 = v18;
  v36 = (unsigned int)(*(_DWORD *)(v18 + 68) - 1);
  if ( *(_WORD *)(*(_QWORD *)(v18 + 48) + 16 * v36) == 262 )
  {
    for ( i = v18; ; i = v35 )
    {
      v67 = v36;
      v37 = *(_QWORD *)(i + 56);
      v66 = i;
      if ( !v37 )
        break;
      while ( 1 )
      {
        v35 = *(_QWORD *)(v37 + 16);
        v63 = v37;
        if ( (unsigned __int8)sub_33CF900(&v66, v35) )
          break;
        v37 = *(_QWORD *)(v63 + 32);
        if ( !v37 )
          goto LABEL_42;
      }
      *(_DWORD *)(i + 36) = *(_DWORD *)(v31 + 200);
      v38 = *(_DWORD *)(v35 + 24);
      if ( v38 < 0 && *(char *)(*(_QWORD *)(a1[2] + 8) - 40LL * (unsigned int)~v38 + 24) < 0 )
        *(_BYTE *)(v31 + 248) |= 2u;
      v36 = (unsigned int)(*(_DWORD *)(v35 + 68) - 1);
      if ( *(_WORD *)(*(_QWORD *)(v35 + 48) + 16 * v36) != 262 )
        goto LABEL_43;
    }
LABEL_42:
    v35 = i;
  }
LABEL_43:
  if ( (*(_BYTE *)(v31 + 248) & 2) != 0 )
  {
    v56 = (unsigned int)v69;
    v57 = (unsigned int)v69 + 1LL;
    if ( v57 > HIDWORD(v69) )
    {
      sub_C8D5F0((__int64)&v68, v70, v57, 8u, v28, v29);
      v56 = (unsigned int)v69;
    }
    *(_QWORD *)&v68[8 * v56] = v31;
    LODWORD(v69) = v69 + 1;
  }
  if ( *(_DWORD *)(v18 + 24) == 2 )
    *(_BYTE *)(v31 + 249) |= 0x10u;
  v39 = *(_DWORD *)(v31 + 200);
  *(_QWORD *)v31 = v35;
  *(_BYTE *)(v31 + 254) |= 4u;
  *(_DWORD *)(v35 + 36) = v39;
  sub_335E4B0((__int64)a1, v31);
  (*(void (__fastcall **)(unsigned __int64 *, __int64))(*a1 + 72))(a1, v31);
  v16 = v78;
  if ( (_DWORD)v78 )
    goto LABEL_20;
LABEL_47:
  v40 = v68;
  v41 = v69;
LABEL_48:
  if ( v41 )
  {
LABEL_49:
    v42 = &v40[8 * v41];
    while ( 2 )
    {
      v43 = (__int64 *)*((_QWORD *)v42 - 1);
      LODWORD(v69) = --v41;
      v44 = *v43;
      while ( v44 )
      {
        if ( *(_DWORD *)(v44 + 24) == 49 )
        {
          v45 = *(_QWORD *)(*(_QWORD *)(v44 + 40) + 80LL);
          v46 = *(_DWORD *)(v45 + 24);
          v47 = (0x3FF8000FFE42uLL >> v46) & 1;
          if ( v46 >= 0x2E )
            LOBYTE(v47) = 0;
          if ( v46 != 324 && !(_BYTE)v47 )
          {
            v48 = a1[6] + ((__int64)*(int *)(v45 + 36) << 8);
            *(_BYTE *)(v48 + 248) |= 4u;
          }
        }
        v49 = *(_DWORD *)(v44 + 64);
        if ( !v49 )
          goto LABEL_48;
        v50 = (unsigned int *)(*(_QWORD *)(v44 + 40) + 40LL * (unsigned int)(v49 - 1));
        v44 = *(_QWORD *)v50;
        if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v50 + 48LL) + 16LL * v50[2]) != 262 )
        {
          if ( v41 )
            goto LABEL_49;
          goto LABEL_61;
        }
      }
      v42 -= 8;
      if ( v41 )
        continue;
      break;
    }
  }
LABEL_61:
  if ( v40 != v70 )
    _libc_free((unsigned __int64)v40);
  if ( !v75 )
    _libc_free((unsigned __int64)v72);
  if ( v77 != v79 )
    _libc_free((unsigned __int64)v77);
}
