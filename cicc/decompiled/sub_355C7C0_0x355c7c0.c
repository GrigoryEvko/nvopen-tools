// Function: sub_355C7C0
// Address: 0x355c7c0
//
__int64 __fastcall sub_355C7C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  signed int v7; // r12d
  unsigned __int64 v8; // rbx
  __int64 v9; // r14
  _QWORD *v10; // rax
  _QWORD *v11; // rcx
  _QWORD *v12; // rdx
  _QWORD *v13; // rsi
  _QWORD *v14; // r9
  _QWORD *v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rax
  signed int v19; // ecx
  __int64 v20; // r13
  __int64 v21; // r12
  __int64 v22; // r14
  __int64 v23; // rbx
  int v24; // r15d
  _QWORD *v25; // rax
  __int64 v26; // r9
  unsigned __int64 v27; // rdx
  __int64 v28; // rsi
  __int64 v29; // rcx
  __int64 v30; // rax
  _QWORD *v31; // rdx
  signed int v32; // ecx
  __int64 v33; // r13
  __int64 v34; // r12
  __int64 v35; // r14
  __int64 v36; // rbx
  signed int v37; // r15d
  unsigned __int64 v38; // r9
  __int64 v39; // r10
  _QWORD *v40; // rax
  __int64 v41; // rsi
  __int64 v42; // rcx
  __int64 v43; // rax
  __int64 v44; // r9
  __int64 v45; // rsi
  __int64 v46; // rax
  __int64 v47; // rax
  int v48; // r13d
  bool v49; // zf
  __int64 v51; // r9
  __int64 v52; // rdx
  __int64 v53; // rax
  _QWORD *v54; // rax
  __int64 v55; // rsi
  __int64 v56; // rdi
  __int64 v57; // rdx
  __m128i *v58; // rax
  int *v59; // rax
  signed int v60; // ecx
  unsigned __int64 *v61; // r13
  unsigned __int64 *v62; // rax
  __int64 v63; // rax
  unsigned __int64 v64; // rdx
  __int64 v65; // rax
  unsigned __int64 *v66; // rax
  unsigned __int64 *v67; // rdx
  unsigned __int64 v68; // rax
  __int64 v69; // rdx
  unsigned __int64 v70; // [rsp+0h] [rbp-D0h]
  unsigned __int64 v71; // [rsp+0h] [rbp-D0h]
  signed int v72; // [rsp+0h] [rbp-D0h]
  signed int v73; // [rsp+0h] [rbp-D0h]
  int v74; // [rsp+8h] [rbp-C8h]
  signed int v75; // [rsp+8h] [rbp-C8h]
  signed int v76; // [rsp+8h] [rbp-C8h]
  signed int v77; // [rsp+8h] [rbp-C8h]
  signed int v78; // [rsp+8h] [rbp-C8h]
  signed int v79; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v80; // [rsp+8h] [rbp-C8h]
  __int64 v82; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v83; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int64 *v84; // [rsp+38h] [rbp-98h] BYREF
  char v85[8]; // [rsp+40h] [rbp-90h] BYREF
  _QWORD *v86; // [rsp+48h] [rbp-88h]
  int v87; // [rsp+54h] [rbp-7Ch]
  char v88; // [rsp+5Ch] [rbp-74h]

  v6 = a1;
  v7 = 0x80000000;
  sub_3545EB0((__int64)v85, a1, (_QWORD *)a2, a3, a5, a6);
  v8 = *(_QWORD *)(a2 + 48);
  v82 = *(_QWORD *)(a2 + 56);
  if ( v8 != v82 )
  {
    v9 = a1 + 40;
    while ( 1 )
    {
      if ( (*(_BYTE *)(v8 + 254) & 8) == 0 || !*(_QWORD *)v8 )
        goto LABEL_60;
      if ( !v88 )
        break;
      v10 = v86;
      v11 = *(_QWORD **)(v6 + 48);
      v12 = &v86[v87];
      v13 = v11;
      if ( v86 == v12 )
      {
LABEL_80:
        v83 = v8;
        if ( v13 )
          goto LABEL_66;
        v51 = v9;
LABEL_72:
        v84 = &v83;
        v51 = sub_3549E00((_QWORD *)(v6 + 32), v51, &v84);
LABEL_73:
        if ( v7 < *(_DWORD *)(v51 + 40) )
          v7 = *(_DWORD *)(v51 + 40);
        v8 += 256LL;
        if ( v82 == v8 )
          goto LABEL_61;
      }
      else
      {
        while ( *v10 != v8 )
        {
          if ( v12 == ++v10 )
            goto LABEL_80;
        }
LABEL_10:
        v13 = v11;
        if ( v11 )
        {
          v14 = (_QWORD *)v9;
          v15 = v11;
          do
          {
            while ( 1 )
            {
              v16 = v15[2];
              v17 = v15[3];
              if ( v15[4] >= v8 )
                break;
              v15 = (_QWORD *)v15[3];
              if ( !v17 )
                goto LABEL_15;
            }
            v14 = v15;
            v15 = (_QWORD *)v15[2];
          }
          while ( v16 );
LABEL_15:
          if ( v14 != (_QWORD *)v9
            && v14[4] <= v8
            && !((*((_DWORD *)v14 + 10) - *(_DWORD *)(v6 + 80)) / *(_DWORD *)(v6 + 88)) )
          {
            v83 = v8;
LABEL_66:
            v51 = v9;
            do
            {
              while ( 1 )
              {
                v52 = v13[2];
                v53 = v13[3];
                if ( v13[4] >= v8 )
                  break;
                v13 = (_QWORD *)v13[3];
                if ( !v53 )
                  goto LABEL_70;
              }
              v51 = (__int64)v13;
              v13 = (_QWORD *)v13[2];
            }
            while ( v52 );
LABEL_70:
            if ( v51 != v9 && *(_QWORD *)(v51 + 32) <= v8 )
              goto LABEL_73;
            goto LABEL_72;
          }
        }
        v74 = *(_DWORD *)(v6 + 80);
        v18 = sub_35459D0(*(_QWORD **)(a2 + 3464), v8);
        v19 = v74;
        v20 = *(_QWORD *)v18;
        if ( *(_QWORD *)v18 + 32LL * *(unsigned int *)(v18 + 8) != *(_QWORD *)v18 )
        {
          v75 = v7;
          v21 = v9;
          v22 = *(_QWORD *)v18 + 32LL * *(unsigned int *)(v18 + 8);
          v70 = v8;
          v23 = v6;
          v24 = v19;
          do
          {
            while ( *(_DWORD *)(v20 + 24) )
            {
              v20 += 32;
              if ( v22 == v20 )
                goto LABEL_32;
            }
            v25 = *(_QWORD **)(v23 + 48);
            v26 = v21;
            v27 = *(_QWORD *)(v20 + 8) & 0xFFFFFFFFFFFFFFF8LL;
            v83 = v27;
            if ( !v25 )
              goto LABEL_28;
            do
            {
              while ( 1 )
              {
                v28 = v25[2];
                v29 = v25[3];
                if ( v25[4] >= v27 )
                  break;
                v25 = (_QWORD *)v25[3];
                if ( !v29 )
                  goto LABEL_26;
              }
              v26 = (__int64)v25;
              v25 = (_QWORD *)v25[2];
            }
            while ( v28 );
LABEL_26:
            if ( v26 == v21 || *(_QWORD *)(v26 + 32) > v27 )
            {
LABEL_28:
              v84 = &v83;
              v26 = sub_3549E00((_QWORD *)(v23 + 32), v26, &v84);
            }
            if ( v24 < *(_DWORD *)(v26 + 40) )
              v24 = *(_DWORD *)(v26 + 40);
            v20 += 32;
          }
          while ( v22 != v20 );
LABEL_32:
          v19 = v24;
          v9 = v21;
          v6 = v23;
          v7 = v75;
          v8 = v70;
        }
        v76 = v19;
        v30 = sub_3545E90(*(_QWORD **)(a2 + 3464), v8);
        v31 = *(_QWORD **)(v6 + 48);
        v32 = v76;
        v33 = *(_QWORD *)v30;
        if ( *(_QWORD *)v30 + 32LL * *(unsigned int *)(v30 + 8) != *(_QWORD *)v30 )
        {
          v77 = v7;
          v34 = v9;
          v35 = *(_QWORD *)v30 + 32LL * *(unsigned int *)(v30 + 8);
          v71 = v8;
          v36 = v6;
          v37 = v32;
          do
          {
            while ( *(_DWORD *)(v33 + 24) != 1 )
            {
              v33 += 32;
              if ( v35 == v33 )
                goto LABEL_48;
            }
            v38 = *(_QWORD *)v33;
            v39 = v34;
            v83 = *(_QWORD *)v33;
            if ( !v31 )
              goto LABEL_44;
            v40 = v31;
            do
            {
              while ( 1 )
              {
                v41 = v40[2];
                v42 = v40[3];
                if ( v40[4] >= v38 )
                  break;
                v40 = (_QWORD *)v40[3];
                if ( !v42 )
                  goto LABEL_42;
              }
              v39 = (__int64)v40;
              v40 = (_QWORD *)v40[2];
            }
            while ( v41 );
LABEL_42:
            if ( v39 == v34 || *(_QWORD *)(v39 + 32) > v38 )
            {
LABEL_44:
              v84 = &v83;
              v43 = sub_3549E00((_QWORD *)(v36 + 32), v39, &v84);
              v31 = *(_QWORD **)(v36 + 48);
              v39 = v43;
            }
            if ( v37 < *(_DWORD *)(v39 + 40) )
              v37 = *(_DWORD *)(v39 + 40);
            v33 += 32;
          }
          while ( v35 != v33 );
LABEL_48:
          v32 = v37;
          v9 = v34;
          v6 = v36;
          v7 = v77;
          v8 = v71;
        }
        v83 = v8;
        v44 = v9;
        if ( !v31 )
          goto LABEL_56;
        do
        {
          while ( 1 )
          {
            v45 = v31[2];
            v46 = v31[3];
            if ( v31[4] >= v8 )
              break;
            v31 = (_QWORD *)v31[3];
            if ( !v46 )
              goto LABEL_54;
          }
          v44 = (__int64)v31;
          v31 = (_QWORD *)v31[2];
        }
        while ( v45 );
LABEL_54:
        if ( v44 == v9 || *(_QWORD *)(v44 + 32) > v8 )
        {
LABEL_56:
          v78 = v32;
          v84 = &v83;
          v47 = sub_3549E00((_QWORD *)(v6 + 32), v44, &v84);
          v32 = v78;
          v44 = v47;
        }
        v48 = *(_DWORD *)(v44 + 40);
        if ( v48 != v32 )
        {
          v54 = *(_QWORD **)(v6 + 48);
          v83 = v8;
          v55 = v9;
          if ( !v54 )
            goto LABEL_95;
          do
          {
            while ( 1 )
            {
              v56 = v54[2];
              v57 = v54[3];
              if ( v54[4] >= v8 )
                break;
              v54 = (_QWORD *)v54[3];
              if ( !v57 )
                goto LABEL_87;
            }
            v55 = (__int64)v54;
            v54 = (_QWORD *)v54[2];
          }
          while ( v56 );
LABEL_87:
          if ( v55 == v9 || *(_QWORD *)(v55 + 32) > v8 )
          {
LABEL_95:
            v72 = v32;
            v84 = &v83;
            v63 = sub_3549E00((_QWORD *)(v6 + 32), v55, &v84);
            v32 = v72;
            v55 = v63;
          }
          *(_DWORD *)(v55 + 40) = v32;
          v79 = v32;
          LODWORD(v84) = v48;
          v58 = (__m128i *)sub_354BE50(v6, (int *)&v84);
          sub_355C6D0(v58, v8);
          LODWORD(v84) = v79;
          v59 = sub_354BE50(v6, (int *)&v84);
          v60 = v79;
          v61 = (unsigned __int64 *)v59;
          v62 = (unsigned __int64 *)*((_QWORD *)v59 + 6);
          if ( v62 != (unsigned __int64 *)(v61[8] - 8) )
          {
            if ( v62 )
            {
              *v62 = v8;
              v62 = (unsigned __int64 *)v61[6];
            }
            v61[6] = (unsigned __int64)(v62 + 1);
            if ( v7 < v79 )
              v7 = v79;
            goto LABEL_60;
          }
          v64 = v61[9];
          if ( ((__int64)(v61[4] - v61[2]) >> 3)
             + ((((__int64)(v64 - v61[5]) >> 3) - 1) << 6)
             + ((__int64)((__int64)v62 - v61[7]) >> 3) == 0xFFFFFFFFFFFFFFFLL )
            sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
          if ( v61[1] - ((__int64)(v64 - *v61) >> 3) <= 1 )
          {
            sub_354AE70(v61, 1u, 0);
            v64 = v61[9];
            v60 = v79;
          }
          v73 = v60;
          v80 = v64;
          v65 = sub_22077B0(0x200u);
          v32 = v73;
          *(_QWORD *)(v80 + 8) = v65;
          v66 = (unsigned __int64 *)v61[6];
          if ( v66 )
            *v66 = v8;
          v67 = (unsigned __int64 *)(v61[9] + 8);
          v61[9] = (unsigned __int64)v67;
          v68 = *v67;
          v69 = *v67 + 512;
          v61[7] = v68;
          v61[8] = v69;
          v61[6] = v68;
        }
        if ( v7 < v32 )
          v7 = v32;
LABEL_60:
        v8 += 256LL;
        if ( v82 == v8 )
          goto LABEL_61;
      }
    }
    if ( !sub_C8CA60((__int64)v85, v8) )
    {
      v13 = *(_QWORD **)(v6 + 48);
      goto LABEL_80;
    }
    v11 = *(_QWORD **)(v6 + 48);
    goto LABEL_10;
  }
LABEL_61:
  v49 = v88 == 0;
  *(_DWORD *)(v6 + 84) = v7;
  if ( v49 )
    _libc_free((unsigned __int64)v86);
  return 1;
}
