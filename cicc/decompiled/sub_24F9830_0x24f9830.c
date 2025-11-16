// Function: sub_24F9830
// Address: 0x24f9830
//
__int64 __fastcall sub_24F9830(__int64 *a1, __int64 *a2)
{
  _QWORD *v3; // rax
  __int64 v4; // rbx
  __int64 v5; // rax
  signed __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // rcx
  _QWORD *v10; // rax
  unsigned __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rdx
  __int64 v14; // r9
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rbx
  __int64 *v19; // rax
  __int64 v20; // r14
  __int64 *v21; // r13
  _QWORD *v22; // rax
  __int64 v23; // r9
  unsigned int v24; // eax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rsi
  _QWORD *v28; // rdx
  __int64 v29; // rcx
  unsigned int v30; // eax
  __int64 v31; // rax
  __int64 v32; // rsi
  char *v33; // rdi
  char *v34; // rdx
  _BYTE *v35; // r12
  size_t v36; // rdx
  int v37; // eax
  size_t v38; // rdx
  int v39; // eax
  bool v40; // zf
  char v41; // al
  __int64 *v42; // rax
  int v43; // ecx
  unsigned __int64 v44; // rdx
  unsigned int v45; // eax
  int v46; // ecx
  unsigned __int64 v47; // rdx
  unsigned __int64 v48; // r11
  int v49; // eax
  __int64 v50; // r10
  __int64 v51; // r10
  size_t v52; // rdx
  int v53; // [rsp+0h] [rbp-130h]
  int v54; // [rsp+0h] [rbp-130h]
  __int64 v55; // [rsp+0h] [rbp-130h]
  __int64 v56; // [rsp+0h] [rbp-130h]
  __int64 v57; // [rsp+8h] [rbp-128h]
  __int64 v58; // [rsp+8h] [rbp-128h]
  __int64 v59; // [rsp+8h] [rbp-128h]
  __int64 v60; // [rsp+8h] [rbp-128h]
  unsigned __int8 v61; // [rsp+27h] [rbp-109h]
  __int64 v62; // [rsp+28h] [rbp-108h]
  __int64 v63; // [rsp+30h] [rbp-100h]
  __int64 v64; // [rsp+38h] [rbp-F8h]
  __int64 v65; // [rsp+40h] [rbp-F0h]
  __int64 v66; // [rsp+48h] [rbp-E8h]
  unsigned __int8 v67; // [rsp+48h] [rbp-E8h]
  char v68; // [rsp+48h] [rbp-E8h]
  char v69; // [rsp+48h] [rbp-E8h]
  char v70; // [rsp+48h] [rbp-E8h]
  unsigned __int8 v71; // [rsp+48h] [rbp-E8h]
  unsigned __int8 v72; // [rsp+48h] [rbp-E8h]
  char v73; // [rsp+48h] [rbp-E8h]
  __int64 v74; // [rsp+58h] [rbp-D8h] BYREF
  void *v75[2]; // [rsp+60h] [rbp-D0h] BYREF
  _BYTE v76[48]; // [rsp+70h] [rbp-C0h] BYREF
  int v77; // [rsp+A0h] [rbp-90h]
  void *s2[2]; // [rsp+B0h] [rbp-80h] BYREF
  _BYTE v79[48]; // [rsp+C0h] [rbp-70h] BYREF
  int v80; // [rsp+F0h] [rbp-40h]

  v61 = 0;
  v63 = *a2;
  v65 = *a2 + 8LL * *((unsigned int *)a2 + 2);
  if ( v65 != *a2 )
  {
    do
    {
      s2[0] = *(void **)(v65 - 8);
      v3 = sub_24F9690((__int64)a1, s2);
      v4 = *a1;
      v5 = (__int64)v3 - *a1;
      v66 = a1[34];
      v62 = v5 >> 3;
      v6 = 8 * (v62 + 2 * (v62 + (v5 & 0xFFFFFFFFFFFFFFF8LL)));
      v7 = v66 + v6;
      v8 = *(_QWORD *)(*(_QWORD *)(*a1 + 5425221848LL * (unsigned int)(v6 >> 3)) + 16LL);
      if ( v8 )
      {
        while ( 1 )
        {
          v9 = *(_QWORD *)(v8 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v9 - 30) <= 0xAu )
            break;
          v8 = *(_QWORD *)(v8 + 8);
          if ( !v8 )
            goto LABEL_8;
        }
LABEL_6:
        s2[0] = *(void **)(v9 + 40);
        v10 = sub_24F9690((__int64)a1, s2);
        v13 = ((__int64)v10 - v4) >> 3;
        v14 = *(unsigned __int8 *)(v66
                                 + 8 * (v13 + 2 * (v13 + (((unsigned __int64)v10 - v4) & 0xFFFFFFFFFFFFFFF8LL)))
                                 + 147);
        if ( (_BYTE)v14 )
        {
          v75[0] = v76;
          v75[1] = (void *)0x600000000LL;
          if ( *(_DWORD *)(v7 + 8) )
          {
            v72 = v14;
            sub_24F9330((__int64)v75, v7, v13, v11, v12, v14);
            v14 = v72;
          }
          v77 = *(_DWORD *)(v7 + 64);
          s2[0] = v79;
          s2[1] = (void *)0x600000000LL;
          v16 = *(unsigned int *)(v7 + 80);
          if ( (_DWORD)v16 )
          {
            v71 = v14;
            sub_24F9330((__int64)s2, v7 + 72, v16, v11, v12, v14);
            v14 = v71;
          }
          v17 = *a1;
          v80 = *(_DWORD *)(v7 + 136);
          v18 = *(_QWORD *)(*(_QWORD *)(v17 + 5425221848LL * (unsigned int)((v7 - a1[34]) >> 3)) + 16LL);
          if ( v18 )
          {
            while ( 1 )
            {
              v17 = *(_QWORD *)(v18 + 24);
              if ( (unsigned __int8)(*(_BYTE *)v17 - 30) <= 0xAu )
                break;
              v18 = *(_QWORD *)(v18 + 8);
              if ( !v18 )
                goto LABEL_32;
            }
            v67 = v14;
            v64 = v7 + 72;
            v19 = a1;
            v20 = v7;
            v21 = v19;
LABEL_20:
            v74 = *(_QWORD *)(v17 + 40);
            v22 = sub_24F9690((__int64)v21, &v74);
            v23 = v21[34]
                + 8
                * ((((__int64)v22 - *v21) >> 3)
                 + 2 * ((((__int64)v22 - *v21) >> 3) + (((unsigned __int64)v22 - *v21) & 0xFFFFFFFFFFFFFFF8LL)));
            v24 = *(_DWORD *)(v23 + 64);
            if ( *(_DWORD *)(v20 + 64) < v24 )
            {
              v46 = *(_DWORD *)(v20 + 64) & 0x3F;
              if ( v46 )
                *(_QWORD *)(*(_QWORD *)v20 + 8LL * *(unsigned int *)(v20 + 8) - 8) &= ~(-1LL << v46);
              v47 = *(unsigned int *)(v20 + 8);
              *(_DWORD *)(v20 + 64) = v24;
              v48 = (v24 + 63) >> 6;
              if ( v48 != v47 )
              {
                if ( v48 >= v47 )
                {
                  v51 = v48 - v47;
                  if ( v48 > *(unsigned int *)(v20 + 12) )
                  {
                    v55 = v48 - v47;
                    v59 = v23;
                    sub_C8D5F0(v20, (const void *)(v20 + 16), v48, 8u, v12, v23);
                    v47 = *(unsigned int *)(v20 + 8);
                    v51 = v55;
                    v23 = v59;
                  }
                  if ( 8 * v51 )
                  {
                    v54 = v51;
                    v58 = v23;
                    memset((void *)(*(_QWORD *)v20 + 8 * v47), 0, 8 * v51);
                    LODWORD(v47) = *(_DWORD *)(v20 + 8);
                    LODWORD(v51) = v54;
                    v23 = v58;
                  }
                  v24 = *(_DWORD *)(v20 + 64);
                  *(_DWORD *)(v20 + 8) = v51 + v47;
                }
                else
                {
                  *(_DWORD *)(v20 + 8) = (v24 + 63) >> 6;
                }
              }
              v49 = v24 & 0x3F;
              if ( v49 )
                *(_QWORD *)(*(_QWORD *)v20 + 8LL * *(unsigned int *)(v20 + 8) - 8) &= ~(-1LL << v49);
            }
            v25 = 0;
            v26 = *(unsigned int *)(v23 + 8);
            v27 = 8 * v26;
            if ( (_DWORD)v26 )
            {
              do
              {
                v28 = (_QWORD *)(v25 + *(_QWORD *)v20);
                v29 = *(_QWORD *)(*(_QWORD *)v23 + v25);
                v25 += 8;
                *v28 |= v29;
              }
              while ( v27 != v25 );
            }
            v11 = *(unsigned int *)(v20 + 136);
            v30 = *(_DWORD *)(v23 + 136);
            if ( (unsigned int)v11 < v30 )
            {
              v43 = *(_DWORD *)(v20 + 136) & 0x3F;
              if ( v43 )
                *(_QWORD *)(*(_QWORD *)(v20 + 72) + 8LL * *(unsigned int *)(v20 + 80) - 8) &= ~(-1LL << v43);
              v44 = *(unsigned int *)(v20 + 80);
              *(_DWORD *)(v20 + 136) = v30;
              v11 = (v30 + 63) >> 6;
              if ( v11 != v44 )
              {
                if ( v11 >= v44 )
                {
                  v50 = v11 - v44;
                  if ( v11 > *(unsigned int *)(v20 + 84) )
                  {
                    v56 = v11 - v44;
                    v60 = v23;
                    sub_C8D5F0(v64, (const void *)(v20 + 88), v11, 8u, v12, v23);
                    v44 = *(unsigned int *)(v20 + 80);
                    v50 = v56;
                    v23 = v60;
                  }
                  if ( 8 * v50 )
                  {
                    v53 = v50;
                    v57 = v23;
                    memset((void *)(*(_QWORD *)(v20 + 72) + 8 * v44), 0, 8 * v50);
                    LODWORD(v44) = *(_DWORD *)(v20 + 80);
                    LODWORD(v50) = v53;
                    v23 = v57;
                  }
                  v30 = *(_DWORD *)(v20 + 136);
                  *(_DWORD *)(v20 + 80) = v50 + v44;
                }
                else
                {
                  *(_DWORD *)(v20 + 80) = v11;
                }
              }
              v45 = v30 & 0x3F;
              if ( v45 )
              {
                v11 = v45;
                *(_QWORD *)(*(_QWORD *)(v20 + 72) + 8LL * *(unsigned int *)(v20 + 80) - 8) &= ~(-1LL << v45);
              }
            }
            v31 = 0;
            v17 = *(unsigned int *)(v23 + 80);
            v32 = 8 * v17;
            if ( (_DWORD)v17 )
            {
              do
              {
                v17 = v31 + *(_QWORD *)(v20 + 72);
                v11 = *(_QWORD *)(*(_QWORD *)(v23 + 72) + v31);
                v31 += 8;
                *(_QWORD *)v17 |= v11;
              }
              while ( v32 != v31 );
            }
            if ( *(_BYTE *)(v23 + 144) )
              sub_24F9570(v64, v23, v17, v11, v12, v23);
            while ( 1 )
            {
              v18 = *(_QWORD *)(v18 + 8);
              if ( !v18 )
                break;
              v17 = *(_QWORD *)(v18 + 24);
              if ( (unsigned __int8)(*(_BYTE *)v17 - 30) <= 0xAu )
                goto LABEL_20;
            }
            v42 = v21;
            v7 = v20;
            v14 = v67;
            v40 = *(_BYTE *)(v20 + 144) == 0;
            a1 = v42;
            if ( !v40 )
            {
LABEL_51:
              v70 = v14;
              sub_24F9570(v7 + 72, v7, v17, v11, v12, v14);
              LOBYTE(v14) = v70;
LABEL_35:
              v35 = s2[0];
              if ( *(_DWORD *)(v7 + 136) != v80
                || (v36 = 8LL * *(unsigned int *)(v7 + 80)) != 0
                && (v68 = v14, v37 = memcmp(*(const void **)(v7 + 72), s2[0], v36), LOBYTE(v14) = v68, v37)
                || *(_DWORD *)(v7 + 64) != v77 )
              {
                v61 = v14;
              }
              else
              {
                v69 = v14;
                v38 = 8LL * *(unsigned int *)(v7 + 8);
                if ( v38 )
                {
                  v39 = memcmp(*(const void **)v7, v75[0], v38);
                  LOBYTE(v14) = v69;
                  v40 = v39 == 0;
                  v41 = v61;
                  if ( !v40 )
                    v41 = v69;
                  v61 = v41;
                  if ( v40 )
                    LOBYTE(v14) = 0;
                }
                else
                {
                  LOBYTE(v14) = 0;
                }
              }
              *(_BYTE *)(v7 + 147) = v14;
              if ( v35 != v79 )
                _libc_free((unsigned __int64)v35);
              if ( v75[0] != v76 )
                _libc_free((unsigned __int64)v75[0]);
              goto LABEL_9;
            }
          }
          else
          {
LABEL_32:
            if ( *(_BYTE *)(v7 + 144) )
              goto LABEL_51;
          }
          v33 = *(char **)(v7 + 72);
          if ( *(_BYTE *)(v7 + 145) )
          {
            v52 = 8LL * *(unsigned int *)(v7 + 80);
            if ( v52 )
            {
              v73 = v14;
              memset(v33, 0, v52);
              LOBYTE(v14) = v73;
            }
          }
          else
          {
            v34 = &v33[8 * ((unsigned int)v62 >> 6)];
            *(_BYTE *)(v7 + 146) |= (*(_QWORD *)v34 >> v62) & 1;
            *(_QWORD *)v34 &= ~(1LL << v62);
          }
          goto LABEL_35;
        }
        while ( 1 )
        {
          v8 = *(_QWORD *)(v8 + 8);
          if ( !v8 )
            break;
          v9 = *(_QWORD *)(v8 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v9 - 30) <= 0xAu )
            goto LABEL_6;
        }
      }
LABEL_8:
      *(_BYTE *)(v7 + 147) = 0;
LABEL_9:
      v65 -= 8;
    }
    while ( v63 != v65 );
  }
  return v61;
}
