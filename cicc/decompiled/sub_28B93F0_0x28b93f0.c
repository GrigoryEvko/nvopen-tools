// Function: sub_28B93F0
// Address: 0x28b93f0
//
void __fastcall sub_28B93F0(_QWORD *a1, char *a2, __int64 a3)
{
  __int64 v3; // rax
  char *v4; // r8
  __int64 v6; // r15
  __int64 v7; // r11
  __int64 *v8; // rdi
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  unsigned int v12; // ecx
  __int64 v13; // rax
  unsigned int v14; // edx
  __int64 v15; // r8
  __int64 v16; // rbx
  __int64 v17; // rax
  unsigned int v18; // esi
  __int64 v19; // rax
  __int64 v20; // r10
  __int64 v21; // rbx
  __int64 v22; // r8
  __int64 v23; // rdi
  __int64 v24; // r9
  __int64 *v25; // r12
  char *v26; // rbx
  char *v27; // r13
  __int64 v28; // rax
  unsigned int v29; // ecx
  __int64 v30; // rax
  unsigned int v31; // edx
  char *v32; // rax
  __int64 v33; // rsi
  unsigned int v34; // ecx
  __int64 v35; // rax
  __int64 v36; // r10
  __int64 v37; // rbx
  __int64 v38; // rax
  unsigned int v39; // esi
  __int64 v40; // rax
  unsigned __int64 *v41; // rbx
  unsigned __int64 v42; // rcx
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // r12
  unsigned __int64 v46; // r15
  unsigned __int64 v47; // rdi
  unsigned __int64 v48; // rdi
  __int64 v49; // rdx
  __int64 v50; // rdx
  char *v51; // r14
  __int64 v52; // r15
  unsigned __int64 *v53; // r12
  unsigned __int64 v54; // rcx
  unsigned __int64 v55; // rdx
  unsigned __int64 v56; // rax
  unsigned __int64 v57; // rbx
  unsigned __int64 v58; // r13
  unsigned __int64 v59; // rdi
  unsigned __int64 v60; // rdi
  __int64 *v61; // [rsp+8h] [rbp-68h]
  __int64 v62; // [rsp+8h] [rbp-68h]
  __int64 v63; // [rsp+10h] [rbp-60h]
  char *v64; // [rsp+18h] [rbp-58h]
  __int64 v65; // [rsp+18h] [rbp-58h]
  unsigned __int64 v66; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int64 v67; // [rsp+28h] [rbp-48h]
  unsigned __int64 v68; // [rsp+30h] [rbp-40h]

  v3 = a2 - (char *)a1;
  v63 = a3;
  v64 = a2;
  if ( a2 - (char *)a1 > 384 )
  {
    v4 = (char *)a1;
    if ( a3 )
    {
      v61 = a1 + 6;
      while ( 1 )
      {
        v6 = a1[3];
        v7 = a1[4];
        --v63;
        v8 = &a1[(__int64)(0xAAAAAAAAAAAAAAABLL * ((v64 - (char *)a1) >> 3)) / 2
               + ((0xAAAAAAAAAAAAAAABLL * ((v64 - (char *)a1) >> 3)
                 + ((0xAAAAAAAAAAAAAAABLL * ((v64 - (char *)a1) >> 3)) >> 63))
                & 0xFFFFFFFFFFFFFFFELL)];
        v9 = v8[1];
        v10 = *v8;
        if ( v6 == v7 )
        {
          v12 = -1;
          if ( v10 == v9 )
          {
            v14 = -1;
            v36 = *((_QWORD *)v64 - 3);
            v37 = *((_QWORD *)v64 - 2);
            if ( v37 == v36 )
            {
              v22 = *a1;
              v40 = a1[1];
              v21 = a1[2];
LABEL_78:
              *a1 = v10;
              a1[1] = v8[1];
              a1[2] = v8[2];
              *v8 = v22;
              v8[1] = v40;
LABEL_79:
              v8[2] = v21;
              v22 = a1[3];
              v24 = a1[4];
              v7 = a1[1];
              v6 = *a1;
              v23 = *((_QWORD *)v64 - 3);
              v20 = *((_QWORD *)v64 - 2);
              goto LABEL_23;
            }
            goto LABEL_52;
          }
        }
        else
        {
          v11 = a1[3];
          v12 = -1;
          do
          {
            if ( v12 > *(_DWORD *)(v11 + 92) )
              v12 = *(_DWORD *)(v11 + 92);
            v11 += 192;
          }
          while ( v7 != v11 );
          if ( v10 == v9 )
          {
            v14 = -1;
            goto LABEL_14;
          }
        }
        v13 = *v8;
        v14 = -1;
        do
        {
          if ( v14 > *(_DWORD *)(v13 + 92) )
            v14 = *(_DWORD *)(v13 + 92);
          v13 += 192;
        }
        while ( v13 != v9 );
LABEL_14:
        if ( v12 < v14 )
        {
          v15 = *((_QWORD *)v64 - 3);
          v16 = *((_QWORD *)v64 - 2);
          if ( v16 == v15 )
          {
            v18 = -1;
          }
          else
          {
            v17 = *((_QWORD *)v64 - 3);
            v18 = -1;
            do
            {
              if ( v18 > *(_DWORD *)(v17 + 92) )
                v18 = *(_DWORD *)(v17 + 92);
              v17 += 192;
            }
            while ( v17 != v16 );
          }
          v19 = *a1;
          v20 = a1[1];
          v21 = a1[2];
          if ( v18 <= v14 )
          {
            if ( v12 >= v18 )
            {
              v50 = a1[5];
              a1[5] = v21;
              v24 = v20;
              v22 = v19;
              *a1 = v6;
              a1[1] = v7;
              a1[2] = v50;
              a1[3] = v19;
              a1[4] = v20;
              v23 = *((_QWORD *)v64 - 3);
              v20 = *((_QWORD *)v64 - 2);
            }
            else
            {
              *a1 = v15;
              a1[1] = *((_QWORD *)v64 - 2);
              a1[2] = *((_QWORD *)v64 - 1);
              *((_QWORD *)v64 - 3) = v19;
              *((_QWORD *)v64 - 2) = v20;
              *((_QWORD *)v64 - 1) = v21;
              v22 = a1[3];
              v23 = v19;
              v24 = a1[4];
              v7 = a1[1];
              v6 = *a1;
            }
            goto LABEL_23;
          }
          *a1 = v10;
          a1[1] = v8[1];
          a1[2] = v8[2];
          *v8 = v19;
          v8[1] = v20;
          goto LABEL_79;
        }
        v36 = *((_QWORD *)v64 - 3);
        v37 = *((_QWORD *)v64 - 2);
        if ( v36 == v37 )
        {
          v39 = -1;
          goto LABEL_56;
        }
LABEL_52:
        v38 = v36;
        v39 = -1;
        do
        {
          if ( v39 > *(_DWORD *)(v38 + 92) )
            v39 = *(_DWORD *)(v38 + 92);
          v38 += 192;
        }
        while ( v37 != v38 );
LABEL_56:
        v22 = *a1;
        v40 = a1[1];
        v21 = a1[2];
        if ( v12 < v39 )
        {
          v49 = a1[5];
          a1[5] = v21;
          v24 = v40;
          *a1 = v6;
          a1[1] = v7;
          a1[2] = v49;
          a1[3] = v22;
          a1[4] = v40;
          v23 = *((_QWORD *)v64 - 3);
          v20 = *((_QWORD *)v64 - 2);
          goto LABEL_23;
        }
        if ( v39 <= v14 )
          goto LABEL_78;
        *a1 = v36;
        v20 = v40;
        a1[1] = *((_QWORD *)v64 - 2);
        a1[2] = *((_QWORD *)v64 - 1);
        *((_QWORD *)v64 - 3) = v22;
        *((_QWORD *)v64 - 2) = v40;
        *((_QWORD *)v64 - 1) = v21;
        v23 = v22;
        v24 = a1[4];
        v7 = a1[1];
        v6 = *a1;
        v22 = a1[3];
LABEL_23:
        v25 = v61;
        v26 = v64;
        v27 = (char *)(v61 - 3);
        if ( v24 == v22 )
          goto LABEL_45;
LABEL_24:
        v28 = v22;
        v29 = -1;
        do
        {
          if ( v29 > *(_DWORD *)(v28 + 92) )
            v29 = *(_DWORD *)(v28 + 92);
          v28 += 192;
        }
        while ( v24 != v28 );
        if ( v6 == v7 )
        {
          v31 = -1;
        }
        else
        {
LABEL_29:
          v30 = v6;
          v31 = -1;
          do
          {
            if ( v31 > *(_DWORD *)(v30 + 92) )
              v31 = *(_DWORD *)(v30 + 92);
            v30 += 192;
          }
          while ( v7 != v30 );
        }
        if ( v31 > v29 )
          goto LABEL_44;
        while ( 1 )
        {
          v32 = v26 - 24;
          v26 -= 24;
          if ( v23 == v20 )
          {
LABEL_41:
            v32 -= 24;
            if ( v31 == -1 )
              goto LABEL_42;
            goto LABEL_40;
          }
          while ( 1 )
          {
            v33 = v23;
            v34 = -1;
            do
            {
              if ( v34 > *(_DWORD *)(v33 + 92) )
                v34 = *(_DWORD *)(v33 + 92);
              v33 += 192;
            }
            while ( v20 != v33 );
            v32 -= 24;
            if ( v34 <= v31 )
              break;
LABEL_40:
            v23 = *(_QWORD *)v32;
            v20 = *((_QWORD *)v32 + 1);
            v26 = v32;
            if ( *(_QWORD *)v32 == v20 )
              goto LABEL_41;
          }
LABEL_42:
          if ( v26 <= v27 )
            break;
          *(v25 - 3) = v23;
          v35 = *(v25 - 1);
          *(v25 - 2) = *((_QWORD *)v26 + 1);
          *(v25 - 1) = *((_QWORD *)v26 + 2);
          v23 = *((_QWORD *)v26 - 3);
          *(_QWORD *)v26 = v22;
          v20 = *((_QWORD *)v26 - 2);
          *((_QWORD *)v26 + 1) = v24;
          *((_QWORD *)v26 + 2) = v35;
          v7 = a1[1];
          v6 = *a1;
LABEL_44:
          v22 = *v25;
          v24 = v25[1];
          v25 += 3;
          v27 = (char *)(v25 - 3);
          if ( v24 != v22 )
            goto LABEL_24;
LABEL_45:
          v29 = -1;
          v31 = -1;
          if ( v6 != v7 )
            goto LABEL_29;
        }
        sub_28B93F0(v27, v64, v63);
        v3 = v27 - (char *)a1;
        if ( v27 - (char *)a1 <= 384 )
          return;
        if ( !v63 )
        {
          v64 = v27;
          v4 = (char *)a1;
          break;
        }
        v64 = v27;
      }
    }
    v51 = v4;
    v62 = 0xAAAAAAAAAAAAAAABLL * (v3 >> 3);
    v52 = (v62 - 2) >> 1;
    v53 = (unsigned __int64 *)&v4[8 * v52 + 8 * ((v62 - 2) & 0xFFFFFFFFFFFFFFFELL)];
    while ( 1 )
    {
      v54 = *v53;
      v55 = v53[1];
      v56 = v53[2];
      *v53 = 0;
      v53[2] = 0;
      v53[1] = 0;
      v66 = v54;
      v67 = v55;
      v68 = v56;
      sub_28B7EA0((__int64)v51, v52, v62, &v66);
      v57 = v67;
      v58 = v66;
      if ( v67 != v66 )
      {
        do
        {
          if ( *(_DWORD *)(v58 + 168) > 0x40u )
          {
            v59 = *(_QWORD *)(v58 + 160);
            if ( v59 )
              j_j___libc_free_0_0(v59);
          }
          if ( *(_DWORD *)(v58 + 128) > 0x40u )
          {
            v60 = *(_QWORD *)(v58 + 120);
            if ( v60 )
              j_j___libc_free_0_0(v60);
          }
          if ( (*(_BYTE *)(v58 + 16) & 1) == 0 )
            sub_C7D6A0(*(_QWORD *)(v58 + 24), 8LL * *(unsigned int *)(v58 + 32), 8);
          v58 += 192LL;
        }
        while ( v57 != v58 );
        v58 = v66;
      }
      if ( v58 )
        j_j___libc_free_0(v58);
      v53 -= 3;
      if ( !v52 )
        break;
      --v52;
    }
    v41 = (unsigned __int64 *)(v64 - 24);
    do
    {
      v42 = *v41;
      v43 = v41[1];
      *v41 = 0;
      v41[1] = 0;
      v44 = v41[2];
      v41[2] = 0;
      *v41 = *(_QWORD *)v51;
      v41[1] = *((_QWORD *)v51 + 1);
      v41[2] = *((_QWORD *)v51 + 2);
      v68 = v44;
      v67 = v43;
      v65 = (char *)v41 - v51;
      *(_QWORD *)v51 = 0;
      *((_QWORD *)v51 + 1) = 0;
      *((_QWORD *)v51 + 2) = 0;
      v66 = v42;
      sub_28B7EA0((__int64)v51, 0, 0xAAAAAAAAAAAAAAABLL * (((char *)v41 - v51) >> 3), &v66);
      v45 = v67;
      v46 = v66;
      if ( v67 != v66 )
      {
        do
        {
          if ( *(_DWORD *)(v46 + 168) > 0x40u )
          {
            v47 = *(_QWORD *)(v46 + 160);
            if ( v47 )
              j_j___libc_free_0_0(v47);
          }
          if ( *(_DWORD *)(v46 + 128) > 0x40u )
          {
            v48 = *(_QWORD *)(v46 + 120);
            if ( v48 )
              j_j___libc_free_0_0(v48);
          }
          if ( (*(_BYTE *)(v46 + 16) & 1) == 0 )
            sub_C7D6A0(*(_QWORD *)(v46 + 24), 8LL * *(unsigned int *)(v46 + 32), 8);
          v46 += 192LL;
        }
        while ( v45 != v46 );
        v46 = v66;
      }
      if ( v46 )
        j_j___libc_free_0(v46);
      v41 -= 3;
    }
    while ( v65 > 24 );
  }
}
