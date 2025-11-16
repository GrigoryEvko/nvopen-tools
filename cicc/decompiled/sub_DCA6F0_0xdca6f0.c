// Function: sub_DCA6F0
// Address: 0xdca6f0
//
__int64 __fastcall sub_DCA6F0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6, __int64 *a7)
{
  __int64 v8; // r13
  __int64 v9; // r12
  __int64 v10; // rdx
  unsigned int v11; // ebx
  __int64 v13; // rdx
  __int64 *v14; // r12
  _QWORD *v15; // rax
  __int64 v16; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 *v22; // r15
  __int64 v23; // rax
  __int64 v24; // r15
  bool v26; // zf
  char v27; // al
  int v28; // eax
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // r10
  _QWORD *v34; // rdi
  __int64 v35; // rcx
  _BYTE *v36; // r8
  _BYTE *v37; // rdx
  _BYTE *v38; // rax
  signed __int64 v39; // r11
  int v40; // edx
  __int64 v41; // r9
  const void **v42; // rsi
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 *v45; // rax
  __int64 v46; // r15
  __int64 v47; // rax
  int v48; // eax
  char v49; // al
  _BYTE *v50; // [rsp+0h] [rbp-110h]
  _BYTE *v51; // [rsp+8h] [rbp-108h]
  signed __int64 v52; // [rsp+10h] [rbp-100h]
  __int64 v53; // [rsp+18h] [rbp-F8h]
  __int64 v54; // [rsp+18h] [rbp-F8h]
  int v55; // [rsp+20h] [rbp-F0h]
  __int64 *v56; // [rsp+20h] [rbp-F0h]
  __int64 v57; // [rsp+20h] [rbp-F0h]
  int v58; // [rsp+28h] [rbp-E8h]
  __int64 v59; // [rsp+28h] [rbp-E8h]
  __int64 v60; // [rsp+28h] [rbp-E8h]
  unsigned int v61; // [rsp+30h] [rbp-E0h]
  unsigned int v62; // [rsp+30h] [rbp-E0h]
  int v63; // [rsp+30h] [rbp-E0h]
  __int64 v64; // [rsp+30h] [rbp-E0h]
  unsigned __int8 v67; // [rsp+4Fh] [rbp-C1h]
  const void *v68; // [rsp+50h] [rbp-C0h] BYREF
  unsigned int v69; // [rsp+58h] [rbp-B8h]
  __int64 *v70; // [rsp+60h] [rbp-B0h] BYREF
  const void *v71; // [rsp+68h] [rbp-A8h] BYREF
  unsigned int v72; // [rsp+70h] [rbp-A0h]
  __int64 v73; // [rsp+80h] [rbp-90h] BYREF
  const void *v74; // [rsp+88h] [rbp-88h] BYREF
  __int64 *v75; // [rsp+90h] [rbp-80h]
  char v76; // [rsp+A0h] [rbp-70h]
  _QWORD *v77; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v78; // [rsp+B8h] [rbp-58h]
  _QWORD dest[2]; // [rsp+C0h] [rbp-50h] BYREF
  char v80; // [rsp+D0h] [rbp-40h]

  v8 = a2;
  v9 = a6;
  v10 = *a4;
  if ( !*(_WORD *)(*a4 + 24) )
  {
    v67 = 0;
    v11 = 0;
    while ( 1 )
    {
      v13 = *(_QWORD *)(v10 + 32);
      ++v11;
      v61 = *(_DWORD *)(a6 + 8);
      v14 = (__int64 *)(v13 + 24);
      if ( v61 > 0x40 )
      {
        v59 = v13;
        v28 = sub_C444A0(a6);
        v13 = v59;
        if ( v61 - v28 > 0x40 )
          goto LABEL_6;
        v15 = **(_QWORD ***)a6;
      }
      else
      {
        v15 = *(_QWORD **)a6;
      }
      if ( v15 == (_QWORD *)1 )
      {
        v62 = *(_DWORD *)(a3 + 8);
        if ( v62 > 0x40 )
        {
          v60 = v13;
          v48 = sub_C444A0(a3);
          v13 = v60;
          if ( v62 - v48 <= 0x40 && !**(_QWORD **)a3 )
          {
LABEL_30:
            if ( *(_DWORD *)(v13 + 32) <= 0x40u )
            {
              v49 = v67;
              if ( !*(_QWORD *)(v13 + 24) )
                v49 = 1;
              v67 = v49;
            }
            else
            {
              v63 = *(_DWORD *)(v13 + 32);
              v26 = v63 == (unsigned int)sub_C444A0((__int64)v14);
              v27 = v67;
              if ( v26 )
                v27 = 1;
              v67 = v27;
            }
            goto LABEL_7;
          }
        }
        else if ( !*(_QWORD *)a3 )
        {
          goto LABEL_30;
        }
      }
LABEL_6:
      v67 = 1;
LABEL_7:
      sub_C472A0((__int64)&v77, a6, v14);
      sub_C45EE0(a3, (__int64 *)&v77);
      if ( (unsigned int)v78 > 0x40 && v77 )
        j_j___libc_free_0_0(v77);
      v16 = v11;
      v10 = a4[v11];
      if ( *(_WORD *)(v10 + 24) )
      {
        v9 = a6;
        v8 = a2;
        goto LABEL_12;
      }
    }
  }
  v67 = 0;
  v16 = 0;
  v11 = 0;
LABEL_12:
  if ( v16 != a5 )
  {
    v58 = a3;
    while ( 1 )
    {
      v18 = a4[v16];
      if ( *(_WORD *)(v18 + 24) == 6 )
      {
        v19 = **(_QWORD **)(v18 + 32);
        if ( !*(_WORD *)(v19 + 24) )
          break;
      }
      v73 = v18;
      LODWORD(v75) = *(_DWORD *)(v9 + 8);
      if ( (unsigned int)v75 > 0x40 )
        sub_C43780((__int64)&v74, (const void **)v9);
      else
        v74 = *(const void **)v9;
      sub_DB2560((__int64)&v77, a1, &v73, (__int64)&v74);
      v22 = (__int64 *)dest[0];
      if ( (unsigned int)v75 > 0x40 && v74 )
        j_j___libc_free_0_0(v74);
      if ( v80 )
      {
        v23 = *(unsigned int *)(v8 + 8);
        v24 = *v22;
        if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(v8 + 12) )
        {
          sub_C8D5F0(v8, (const void *)(v8 + 16), v23 + 1, 8u, v20, v21);
          v23 = *(unsigned int *)(v8 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v8 + 8 * v23) = v24;
        ++*(_DWORD *)(v8 + 8);
      }
      else
      {
        sub_C45EE0((__int64)(v22 + 1), (__int64 *)v9);
        v67 = 1;
      }
LABEL_25:
      v16 = ++v11;
      if ( v11 == a5 )
        return v67;
    }
    v64 = v18;
    sub_C472A0((__int64)&v68, v9, (__int64 *)(*(_QWORD *)(v19 + 32) + 24LL));
    v29 = *(_QWORD *)(v64 + 40);
    v30 = *(_QWORD *)(v64 + 32);
    if ( v29 == 2 )
    {
      v31 = *(_QWORD *)(v30 + 8);
      if ( *(_WORD *)(v31 + 24) == 5 )
      {
        v67 |= sub_DCA6F0(a1, v8, v58, *(_QWORD *)(v31 + 32), *(_QWORD *)(v31 + 40), (unsigned int)&v68, (__int64)a7);
LABEL_53:
        if ( v69 > 0x40 && v68 )
          j_j___libc_free_0_0(v68);
        goto LABEL_25;
      }
    }
    v32 = sub_D91800(v30, v29, 1);
    v34 = dest;
    v78 = v35;
    v36 = (_BYTE *)v32;
    v38 = v37;
    v77 = dest;
    v39 = v37 - v36;
    v40 = 0;
    v41 = v39 >> 3;
    if ( (unsigned __int64)v39 > 0x20 )
    {
      v50 = v38;
      v51 = v36;
      v52 = v39;
      v54 = v39 >> 3;
      v57 = v33;
      sub_C8D5F0(v33, dest, v39 >> 3, 8u, (__int64)v36, v41);
      v40 = v78;
      v36 = v51;
      v39 = v52;
      LODWORD(v41) = v54;
      v34 = &v77[(unsigned int)v78];
      v33 = v57;
      if ( v50 == v51 )
        goto LABEL_42;
    }
    else if ( v38 == v36 )
    {
LABEL_42:
      LODWORD(v78) = v40 + v41;
      v70 = sub_DC8BD0(a7, v33, 0, 0);
      v72 = v69;
      if ( v69 > 0x40 )
        sub_C43780((__int64)&v71, &v68);
      else
        v71 = v68;
      v42 = (const void **)a1;
      sub_DB2560((__int64)&v73, a1, (__int64 *)&v70, (__int64)&v71);
      v45 = v75;
      if ( v72 > 0x40 && v71 )
      {
        v56 = v75;
        j_j___libc_free_0_0(v71);
        v45 = v56;
      }
      if ( v76 )
      {
        v46 = *v45;
        v47 = *(unsigned int *)(v8 + 8);
        if ( v47 + 1 > (unsigned __int64)*(unsigned int *)(v8 + 12) )
        {
          v42 = (const void **)(v8 + 16);
          sub_C8D5F0(v8, (const void *)(v8 + 16), v47 + 1, 8u, v43, v44);
          v47 = *(unsigned int *)(v8 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v8 + 8 * v47) = v46;
        ++*(_DWORD *)(v8 + 8);
      }
      else
      {
        v42 = &v68;
        sub_C45EE0((__int64)(v45 + 1), (__int64 *)&v68);
        v67 = 1;
      }
      if ( v77 != dest )
        _libc_free(v77, v42);
      goto LABEL_53;
    }
    v53 = v33;
    v55 = v41;
    memcpy(v34, v36, v39);
    v40 = v78;
    v33 = v53;
    LODWORD(v41) = v55;
    goto LABEL_42;
  }
  return v67;
}
