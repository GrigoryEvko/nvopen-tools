// Function: sub_316BB60
// Address: 0x316bb60
//
__int64 __fastcall sub_316BB60(__int64 *a1, __int64 *a2, __int64 a3, __int64 **a4)
{
  __int64 result; // rax
  __int64 **v5; // r14
  __int64 *v7; // r12
  unsigned __int64 v8; // rbx
  __int64 *v9; // rax
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // r14
  __int64 *i; // r12
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // r12
  int v16; // eax
  __int64 v17; // rbx
  int v18; // eax
  __int64 v19; // r8
  int v20; // eax
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rsi
  bool v26; // cc
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rbx
  unsigned __int64 v29; // rbx
  unsigned int v30; // edx
  unsigned __int64 v31; // rbx
  __int64 v32; // r15
  __int64 *v33; // rbx
  __int64 v34; // r12
  __int64 v35; // rdi
  int v36; // esi
  int v37; // ecx
  unsigned int v38; // edx
  unsigned __int64 v39; // r8
  char v40; // al
  unsigned int v41; // edx
  __int64 v42; // rax
  __int64 v43; // rdx
  unsigned __int64 v44; // rdi
  __int64 *v45; // r12
  __int64 v46; // r13
  __int64 v47; // r10
  int v48; // r8d
  int v49; // ecx
  unsigned int v50; // eax
  unsigned __int64 v51; // r11
  char v52; // r15
  __int64 v53; // rsi
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rdx
  unsigned __int64 v57; // rdi
  __int64 *v58; // [rsp+0h] [rbp-B0h]
  __int64 v59; // [rsp+8h] [rbp-A8h]
  __int64 v60; // [rsp+10h] [rbp-A0h]
  __int64 *v61; // [rsp+18h] [rbp-98h]
  __int64 v62; // [rsp+20h] [rbp-90h]
  unsigned __int64 v63; // [rsp+20h] [rbp-90h]
  int v64; // [rsp+28h] [rbp-88h]
  __int64 v65; // [rsp+28h] [rbp-88h]
  int v66; // [rsp+30h] [rbp-80h]
  __int64 v67; // [rsp+30h] [rbp-80h]
  int v68; // [rsp+30h] [rbp-80h]
  __int64 *v69; // [rsp+38h] [rbp-78h]
  int v70; // [rsp+38h] [rbp-78h]
  __int64 v71; // [rsp+38h] [rbp-78h]
  int v72; // [rsp+38h] [rbp-78h]
  __int64 v73; // [rsp+38h] [rbp-78h]
  __int64 *v74; // [rsp+40h] [rbp-70h]
  char v75; // [rsp+40h] [rbp-70h]
  unsigned int v77; // [rsp+48h] [rbp-68h]
  __int64 v78; // [rsp+48h] [rbp-68h]
  unsigned __int64 v79; // [rsp+50h] [rbp-60h] BYREF
  __int64 v80; // [rsp+58h] [rbp-58h]
  __int64 v81; // [rsp+60h] [rbp-50h]
  int v82; // [rsp+68h] [rbp-48h]
  int v83; // [rsp+6Ch] [rbp-44h]
  unsigned int v84; // [rsp+70h] [rbp-40h]
  char v85; // [rsp+78h] [rbp-38h]

  result = (char *)a2 - (char *)a1;
  v61 = a2;
  v59 = a3;
  if ( (char *)a2 - (char *)a1 <= 768 )
    return result;
  v5 = a4;
  if ( !a3 )
  {
    v74 = a2;
    goto LABEL_38;
  }
  v58 = a1 + 6;
  while ( 2 )
  {
    --v59;
    v7 = &a1[2 * ((0xAAAAAAAAAAAAAAABLL * (result >> 4)) & 0xFFFFFFFFFFFFFFFELL)
           + 2 * ((__int64)(0xAAAAAAAAAAAAAAABLL * (result >> 4)) >> 1)];
    sub_B4CED0((__int64)&v79, a1[6], **a4);
    v8 = v79;
    sub_B4CED0((__int64)&v79, *v7, **a4);
    v9 = *a4;
    if ( v79 >= v8 )
    {
      sub_B4CED0((__int64)&v79, a1[6], *v9);
      v28 = v79;
      sub_B4CED0((__int64)&v79, *(v61 - 6), **a4);
      if ( v79 < v28 )
        goto LABEL_7;
      sub_B4CED0((__int64)&v79, *v7, **a4);
      v29 = v79;
      sub_B4CED0((__int64)&v79, *(v61 - 6), **a4);
      if ( v79 < v29 )
      {
LABEL_31:
        sub_316B9C0(a1, v61 - 6);
        goto LABEL_8;
      }
LABEL_32:
      sub_316B9C0(a1, v7);
      goto LABEL_8;
    }
    sub_B4CED0((__int64)&v79, *v7, *v9);
    v10 = v79;
    sub_B4CED0((__int64)&v79, *(v61 - 6), **a4);
    if ( v79 < v10 )
      goto LABEL_32;
    sub_B4CED0((__int64)&v79, a1[6], **a4);
    v11 = v79;
    sub_B4CED0((__int64)&v79, *(v61 - 6), **a4);
    if ( v79 < v11 )
      goto LABEL_31;
LABEL_7:
    sub_316B9C0(a1, v58);
LABEL_8:
    v12 = (unsigned __int64)v61;
    for ( i = v58; ; i += 6 )
    {
      v74 = i;
      sub_B4CED0((__int64)&v79, *i, **a4);
      v14 = v79;
      sub_B4CED0((__int64)&v79, *a1, **a4);
      if ( v79 < v14 )
        continue;
      v69 = i;
      for ( v12 -= 48LL; ; v12 -= 48LL )
      {
        sub_B4CED0((__int64)&v79, *a1, **a4);
        v15 = v79;
        sub_B4CED0((__int64)&v79, *(_QWORD *)v12, **a4);
        if ( v79 >= v15 )
          break;
      }
      i = v69;
      if ( (unsigned __int64)v69 >= v12 )
        break;
      v16 = *((_DWORD *)v69 + 6);
      ++v69[1];
      *((_DWORD *)v69 + 6) = 0;
      v17 = v69[2];
      v70 = v16;
      v18 = *((_DWORD *)i + 7);
      i[2] = 0;
      v19 = *i;
      v66 = v18;
      v20 = *((_DWORD *)i + 8);
      *((_DWORD *)i + 7) = 0;
      *((_DWORD *)i + 8) = 0;
      v64 = v20;
      LOBYTE(v20) = *((_BYTE *)i + 40);
      *i = *(_QWORD *)v12;
      v62 = v19;
      v75 = v20;
      sub_C7D6A0(0, 0, 8);
      ++i[1];
      *((_DWORD *)i + 8) = 0;
      i[2] = 0;
      *((_DWORD *)i + 6) = 0;
      *((_DWORD *)i + 7) = 0;
      v21 = *(_QWORD *)(v12 + 16);
      ++*(_QWORD *)(v12 + 8);
      v22 = i[2];
      i[2] = v21;
      LODWORD(v21) = *(_DWORD *)(v12 + 24);
      *(_QWORD *)(v12 + 16) = v22;
      LODWORD(v22) = *((_DWORD *)i + 6);
      *((_DWORD *)i + 6) = v21;
      LODWORD(v21) = *(_DWORD *)(v12 + 28);
      *(_DWORD *)(v12 + 24) = v22;
      LODWORD(v22) = *((_DWORD *)i + 7);
      *((_DWORD *)i + 7) = v21;
      LODWORD(v21) = *(_DWORD *)(v12 + 32);
      *(_DWORD *)(v12 + 28) = v22;
      LODWORD(v22) = *((_DWORD *)i + 8);
      *((_DWORD *)i + 8) = v21;
      *(_DWORD *)(v12 + 32) = v22;
      *((_BYTE *)i + 40) = *(_BYTE *)(v12 + 40);
      v23 = *(unsigned int *)(v12 + 32);
      *(_QWORD *)v12 = v62;
      if ( (_DWORD)v23 )
      {
        v24 = *(_QWORD *)(v12 + 16);
        v25 = v24 + 32 * v23;
        do
        {
          while ( 1 )
          {
            if ( *(_QWORD *)v24 != -8192 && *(_QWORD *)v24 != -4096 )
            {
              if ( *(_BYTE *)(v24 + 24) )
              {
                v26 = *(_DWORD *)(v24 + 16) <= 0x40u;
                *(_BYTE *)(v24 + 24) = 0;
                if ( !v26 )
                {
                  v27 = *(_QWORD *)(v24 + 8);
                  if ( v27 )
                    break;
                }
              }
            }
            v24 += 32;
            if ( v25 == v24 )
              goto LABEL_23;
          }
          v60 = v24;
          j_j___libc_free_0_0(v27);
          v24 = v60 + 32;
        }
        while ( v25 != v60 + 32 );
LABEL_23:
        v23 = *(unsigned int *)(v12 + 32);
      }
      sub_C7D6A0(*(_QWORD *)(v12 + 16), 32 * v23, 8);
      ++*(_QWORD *)(v12 + 8);
      *(_QWORD *)(v12 + 16) = v17;
      *(_DWORD *)(v12 + 24) = v70;
      *(_DWORD *)(v12 + 28) = v66;
      *(_DWORD *)(v12 + 32) = v64;
      *(_BYTE *)(v12 + 40) = v75;
      sub_C7D6A0(0, 0, 8);
    }
    sub_316BB60(v69, v61, v59, a4);
    result = (char *)v69 - (char *)a1;
    if ( (char *)v69 - (char *)a1 > 768 )
    {
      if ( v59 )
      {
        v61 = v69;
        continue;
      }
      v5 = a4;
LABEL_38:
      v31 = 0xAAAAAAAAAAAAAAABLL * (result >> 4);
      v32 = (__int64)(v31 - 2) >> 1;
      v33 = &a1[2 * v32 + 2 * ((v31 - 2) & 0xFFFFFFFFFFFFFFFELL)];
      v34 = 0xAAAAAAAAAAAAAAABLL * (result >> 4);
      while ( 1 )
      {
        v35 = v33[2];
        v36 = *((_DWORD *)v33 + 6);
        v33[2] = 0;
        v37 = *((_DWORD *)v33 + 7);
        v38 = *((_DWORD *)v33 + 8);
        *((_DWORD *)v33 + 6) = 0;
        ++v33[1];
        v39 = *v33;
        *((_DWORD *)v33 + 7) = 0;
        v40 = *((_BYTE *)v33 + 40);
        *((_DWORD *)v33 + 8) = 0;
        v81 = v35;
        v79 = v39;
        v82 = v36;
        v83 = v37;
        v84 = v38;
        v80 = 1;
        v85 = v40;
        sub_315F0E0((__int64)a1, v32, v34, (__int64 *)&v79, v5);
        v41 = v84;
        if ( v84 )
        {
          v42 = v81;
          v43 = v81 + 32LL * v84;
          do
          {
            while ( 1 )
            {
              if ( *(_QWORD *)v42 != -8192 && *(_QWORD *)v42 != -4096 )
              {
                if ( *(_BYTE *)(v42 + 24) )
                {
                  v26 = *(_DWORD *)(v42 + 16) <= 0x40u;
                  *(_BYTE *)(v42 + 24) = 0;
                  if ( !v26 )
                  {
                    v44 = *(_QWORD *)(v42 + 8);
                    if ( v44 )
                      break;
                  }
                }
              }
              v42 += 32;
              if ( v43 == v42 )
                goto LABEL_48;
            }
            v67 = v42;
            v71 = v43;
            j_j___libc_free_0_0(v44);
            v43 = v71;
            v42 = v67 + 32;
          }
          while ( v71 != v67 + 32 );
LABEL_48:
          v41 = v84;
        }
        v33 -= 6;
        sub_C7D6A0(v81, 32LL * v41, 8);
        if ( !v32 )
          break;
        --v32;
        sub_C7D6A0(0, 0, 8);
      }
      sub_C7D6A0(0, 0, 8);
      v45 = v74;
      v46 = (__int64)a1;
      do
      {
        v45 -= 6;
        v47 = v45[2];
        v48 = *((_DWORD *)v45 + 6);
        v49 = *((_DWORD *)v45 + 7);
        v50 = *((_DWORD *)v45 + 8);
        v45[2] = 0;
        ++v45[1];
        v51 = *v45;
        *((_DWORD *)v45 + 6) = 0;
        v52 = *((_BYTE *)v45 + 40);
        *((_DWORD *)v45 + 7) = 0;
        *((_DWORD *)v45 + 8) = 0;
        v63 = v51;
        *v45 = *(_QWORD *)v46;
        v65 = v47;
        v68 = v48;
        v72 = v49;
        v77 = v50;
        sub_C7D6A0(0, 0, 8);
        ++v45[1];
        *((_DWORD *)v45 + 8) = 0;
        v45[2] = 0;
        *((_DWORD *)v45 + 6) = 0;
        *((_DWORD *)v45 + 7) = 0;
        v53 = *(_QWORD *)(v46 + 16);
        ++*(_QWORD *)(v46 + 8);
        v54 = v45[2];
        v45[2] = v53;
        LODWORD(v53) = *(_DWORD *)(v46 + 24);
        *(_QWORD *)(v46 + 16) = v54;
        LODWORD(v54) = *((_DWORD *)v45 + 6);
        *((_DWORD *)v45 + 6) = v53;
        LODWORD(v53) = *(_DWORD *)(v46 + 28);
        *(_DWORD *)(v46 + 24) = v54;
        LODWORD(v54) = *((_DWORD *)v45 + 7);
        *((_DWORD *)v45 + 7) = v53;
        LODWORD(v53) = *(_DWORD *)(v46 + 32);
        *(_DWORD *)(v46 + 28) = v54;
        LODWORD(v54) = *((_DWORD *)v45 + 8);
        *((_DWORD *)v45 + 8) = v53;
        *(_DWORD *)(v46 + 32) = v54;
        LOBYTE(v54) = *(_BYTE *)(v46 + 40);
        v79 = v63;
        *((_BYTE *)v45 + 40) = v54;
        v81 = v65;
        v80 = 1;
        v82 = v68;
        v85 = v52;
        v83 = v72;
        v84 = v77;
        sub_315F0E0(v46, 0, 0xAAAAAAAAAAAAAAABLL * (((__int64)v45 - v46) >> 4), (__int64 *)&v79, v5);
        v30 = v84;
        if ( v84 )
        {
          v55 = v81;
          v56 = v81 + 32LL * v84;
          do
          {
            if ( *(_QWORD *)v55 != -8192 && *(_QWORD *)v55 != -4096 )
            {
              if ( *(_BYTE *)(v55 + 24) )
              {
                v26 = *(_DWORD *)(v55 + 16) <= 0x40u;
                *(_BYTE *)(v55 + 24) = 0;
                if ( !v26 )
                {
                  v57 = *(_QWORD *)(v55 + 8);
                  if ( v57 )
                  {
                    v73 = v55;
                    v78 = v56;
                    j_j___libc_free_0_0(v57);
                    v55 = v73;
                    v56 = v78;
                  }
                }
              }
            }
            v55 += 32;
          }
          while ( v56 != v55 );
          v30 = v84;
        }
        sub_C7D6A0(v81, 32LL * v30, 8);
        result = sub_C7D6A0(0, 0, 8);
      }
      while ( (__int64)v45 - v46 > 48 );
    }
    return result;
  }
}
