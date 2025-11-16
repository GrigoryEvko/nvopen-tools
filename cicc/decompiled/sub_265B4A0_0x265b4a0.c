// Function: sub_265B4A0
// Address: 0x265b4a0
//
void __fastcall sub_265B4A0(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __int64 *a4,
        char a5,
        __int64 a6,
        char *a7,
        __int64 a8)
{
  int *v8; // r15
  char v12; // al
  _DWORD *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r15
  int v17; // esi
  __int64 v18; // rax
  __int64 v19; // rcx
  unsigned int v20; // eax
  _DWORD *v21; // rax
  _DWORD *i; // rdx
  __int64 v23; // rax
  _DWORD *v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rax
  unsigned int v27; // esi
  __int64 v28; // rdx
  int v29; // r9d
  unsigned int v30; // r14d
  __int64 *v31; // rax
  __int64 v32; // rdi
  __int64 *v33; // rax
  __int64 v34; // rdi
  bool v35; // zf
  unsigned __int64 v36; // rax
  int v37; // ecx
  unsigned int v38; // esi
  int v39; // edx
  int v40; // edx
  int v41; // eax
  int v42; // edi
  __int64 v43; // [rsp+0h] [rbp-160h]
  __int64 v44; // [rsp+10h] [rbp-150h]
  unsigned __int8 v46; // [rsp+2Fh] [rbp-131h]
  int v47; // [rsp+30h] [rbp-130h] BYREF
  int v48; // [rsp+34h] [rbp-12Ch] BYREF
  __int64 v49; // [rsp+38h] [rbp-128h] BYREF
  __int64 v50; // [rsp+40h] [rbp-120h] BYREF
  _DWORD *v51; // [rsp+48h] [rbp-118h]
  __int64 v52; // [rsp+50h] [rbp-110h] BYREF
  volatile signed __int32 *v53; // [rsp+58h] [rbp-108h]
  __int64 v54; // [rsp+60h] [rbp-100h] BYREF
  _DWORD *v55; // [rsp+68h] [rbp-F8h]
  __int64 v56; // [rsp+70h] [rbp-F0h]
  unsigned int v57; // [rsp+78h] [rbp-E8h]
  unsigned __int64 v58; // [rsp+80h] [rbp-E0h] BYREF
  _DWORD *v59; // [rsp+88h] [rbp-D8h]
  __int64 v60; // [rsp+90h] [rbp-D0h]
  unsigned int v61; // [rsp+98h] [rbp-C8h]
  unsigned __int64 v62[2]; // [rsp+B0h] [rbp-B0h] BYREF
  _BYTE v63[72]; // [rsp+C0h] [rbp-A0h] BYREF
  int v64; // [rsp+108h] [rbp-58h] BYREF
  unsigned __int64 v65; // [rsp+110h] [rbp-50h]
  int *v66; // [rsp+118h] [rbp-48h]
  int *v67; // [rsp+120h] [rbp-40h]
  __int64 v68; // [rsp+128h] [rbp-38h]

  v8 = (int *)(a1 + 344);
  v12 = 1;
  if ( a5 != 4 )
    v12 = a5;
  ++*(_DWORD *)(a1 + 344);
  v46 = v12;
  *(_BYTE *)sub_26509C0(a1 + 128, (int *)(a1 + 344)) = v12;
  if ( !a8 )
    goto LABEL_4;
  v34 = a1 + 160;
  v35 = (unsigned __int8)sub_264A520(a1 + 160, v8, &v58) == 0;
  v36 = v58;
  if ( v35 )
  {
    v37 = *(_DWORD *)(a1 + 176);
    v38 = *(_DWORD *)(a1 + 184);
    v62[0] = v58;
    ++*(_QWORD *)(a1 + 160);
    v39 = v37 + 1;
    if ( 4 * (v37 + 1) >= 3 * v38 )
    {
      v38 *= 2;
    }
    else if ( v38 - *(_DWORD *)(a1 + 180) - v39 > v38 >> 3 )
    {
LABEL_37:
      *(_DWORD *)(a1 + 176) = v39;
      if ( *(_DWORD *)v36 != -1 )
        --*(_DWORD *)(a1 + 180);
      v40 = *(_DWORD *)(a1 + 344);
      *(_QWORD *)(v36 + 8) = 0;
      *(_QWORD *)(v36 + 16) = 0;
      *(_DWORD *)v36 = v40;
      *(_QWORD *)(v36 + 24) = 0;
      goto LABEL_34;
    }
    sub_2650AE0(v34, v38);
    sub_264A520(v34, v8, v62);
    v39 = *(_DWORD *)(a1 + 176) + 1;
    v36 = v62[0];
    goto LABEL_37;
  }
LABEL_34:
  sub_263FF00(v36 + 8, *(char **)(v36 + 8), a7, &a7[16 * a8]);
LABEL_4:
  *(_BYTE *)(a2 + 2) |= v46;
  v62[0] = (unsigned __int64)v63;
  v62[1] = 0x800000000LL;
  v64 = 0;
  v65 = 0;
  v66 = &v64;
  v67 = &v64;
  v68 = 0;
  v50 = sub_2647370(a3, a4);
  v51 = v13;
  while ( 1 )
  {
    sub_1039B70(&v58, *a3, 1);
    if ( v51 == v59 )
      break;
    v49 = sub_1039BF0((__int64)&v50);
    v16 = sub_263DED0(a1, v49);
    if ( !v16 )
    {
      LODWORD(v59) = 0;
      v58 = 0;
      v26 = sub_2648220((_QWORD *)a1, 0, 0, 0, 0);
      v27 = *(_DWORD *)(a1 + 216);
      v14 = v26;
      if ( v27 )
      {
        v28 = v49;
        v15 = *(_QWORD *)(a1 + 200);
        v29 = 1;
        v30 = (v27 - 1) & (((0xBF58476D1CE4E5B9LL * v49) >> 31) ^ (484763065 * v49));
        v31 = (__int64 *)(v15 + 16LL * v30);
        v32 = *v31;
        if ( *v31 == v49 )
        {
LABEL_28:
          v33 = v31 + 1;
LABEL_29:
          *v33 = v14;
          v16 = v14;
          *(_QWORD *)(v14 + 40) = v49;
          goto LABEL_7;
        }
        while ( v32 != -1 )
        {
          if ( !v16 && v32 == -2 )
            v16 = (__int64)v31;
          v30 = (v27 - 1) & (v29 + v30);
          v31 = (__int64 *)(v15 + 16LL * v30);
          v32 = *v31;
          if ( v49 == *v31 )
            goto LABEL_28;
          ++v29;
        }
        if ( !v16 )
          v16 = (__int64)v31;
        v41 = *(_DWORD *)(a1 + 208);
        ++*(_QWORD *)(a1 + 192);
        v42 = v41 + 1;
        v58 = v16;
        if ( 4 * (v41 + 1) < 3 * v27 )
        {
          v15 = v27 >> 3;
          if ( v27 - *(_DWORD *)(a1 + 212) - v42 > (unsigned int)v15 )
          {
LABEL_50:
            *(_DWORD *)(a1 + 208) = v42;
            if ( *(_QWORD *)v16 != -1 )
              --*(_DWORD *)(a1 + 212);
            *(_QWORD *)v16 = v28;
            v33 = (__int64 *)(v16 + 8);
            *(_QWORD *)(v16 + 8) = 0;
            goto LABEL_29;
          }
          v43 = v14;
LABEL_55:
          sub_26451B0(a1 + 192, v27);
          sub_263DF60(a1 + 192, &v49, &v58);
          v28 = v49;
          v16 = v58;
          v14 = v43;
          v42 = *(_DWORD *)(a1 + 208) + 1;
          goto LABEL_50;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 192);
        v58 = 0;
      }
      v43 = v14;
      v27 *= 2;
      goto LABEL_55;
    }
LABEL_7:
    if ( !(_BYTE)qword_4FF3708 )
    {
      sub_265B2C0((__int64)&v58, (__int64)v62, (unsigned __int64 *)&v49, v14, v15);
      if ( !(_BYTE)v60 )
        *(_BYTE *)(v16 + 1) = 1;
    }
    *(_BYTE *)(v16 + 2) |= v46;
    v17 = *(_DWORD *)(a1 + 344);
    v18 = *(_QWORD *)(a2 + 72);
    v19 = *(_QWORD *)(a2 + 80);
    v47 = v17;
    if ( v18 == v19 )
    {
LABEL_14:
      v48 = v17;
      v54 = 0;
      v20 = sub_AF1560(2u);
      v57 = v20;
      if ( v20 )
      {
        v21 = (_DWORD *)sub_C7D670(4LL * v20, 4);
        v56 = 0;
        v55 = v21;
        for ( i = &v21[v57]; i != v21; ++v21 )
        {
          if ( v21 )
            *v21 = -1;
        }
      }
      else
      {
        v55 = 0;
        v56 = 0;
      }
      sub_22B6470((__int64)&v58, (__int64)&v54, &v48);
      v52 = 0;
      v23 = sub_22077B0(0x48u);
      if ( v23 )
      {
        v24 = v55;
        *(_QWORD *)(v23 + 16) = a2;
        *(_QWORD *)(v23 + 24) = v16;
        *(_QWORD *)(v23 + 8) = 0x100000001LL;
        *(_QWORD *)v23 = off_49D3C50;
        v59 = v24;
        v25 = v56;
        *(_WORD *)(v23 + 32) = v46;
        *(_QWORD *)(v23 + 40) = 0;
        *(_QWORD *)(v23 + 48) = 0;
        *(_QWORD *)(v23 + 56) = 0;
        *(_DWORD *)(v23 + 64) = 0;
        v60 = v25;
        v44 = v23;
        v61 = v57;
        ++v54;
        v58 = 1;
        v55 = 0;
        v56 = 0;
        v57 = 0;
        sub_2649AA0(v23 + 40, (__int64)&v58);
        sub_2342640((__int64)&v58);
        v23 = v44;
      }
      v53 = (volatile signed __int32 *)v23;
      v52 = v23 + 16;
      sub_2342640((__int64)&v54);
      sub_2647660((unsigned __int64 *)(a2 + 72), &v52);
      sub_2647660((unsigned __int64 *)(v16 + 48), &v52);
      if ( v53 )
        sub_A191D0(v53);
    }
    else
    {
      while ( v16 != *(_QWORD *)(*(_QWORD *)v18 + 8LL) )
      {
        v18 += 16;
        if ( v19 == v18 )
          goto LABEL_14;
      }
      *(_BYTE *)(*(_QWORD *)v18 + 16LL) |= v46;
      sub_22B6470((__int64)&v58, *(_QWORD *)v18 + 24LL, &v47);
    }
    v51 += 2;
    a2 = v16;
  }
  sub_26414D0(v65);
  if ( (_BYTE *)v62[0] != v63 )
    _libc_free(v62[0]);
}
