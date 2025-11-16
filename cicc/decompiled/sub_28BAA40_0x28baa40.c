// Function: sub_28BAA40
// Address: 0x28baa40
//
unsigned __int64 *__fastcall sub_28BAA40(unsigned __int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  _DWORD *v3; // r14
  unsigned __int64 v4; // rax
  __int64 v5; // r15
  unsigned __int64 *v6; // r14
  unsigned __int64 v7; // r13
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // r10
  __int64 v11; // r11
  __int64 v12; // r8
  __int64 *v13; // rdx
  __int64 *v14; // rax
  unsigned int v15; // eax
  int v16; // eax
  __int64 v17; // r9
  __int64 v18; // rbx
  __int64 v19; // rax
  int v20; // edi
  __int64 v21; // rax
  char *v22; // r12
  char *v23; // r13
  unsigned __int64 v24; // rax
  char *v25; // rbx
  char *v26; // rdi
  __int64 *v28; // rbx
  __int64 *v29; // rdi
  int v30; // eax
  __int64 v31; // r12
  unsigned int v32; // eax
  unsigned __int64 v33; // r12
  bool v34; // bl
  __int64 *v35; // rax
  __int64 *v36; // rdi
  __int64 v37; // r8
  __int64 v38; // rbx
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // rdx
  bool v41; // cf
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // r12
  __int64 v44; // r13
  _QWORD *v45; // rbx
  unsigned __int64 *v46; // r12
  _QWORD *i; // r13
  __int64 v48; // rax
  unsigned __int64 v49; // rbx
  unsigned __int64 v50; // r15
  unsigned __int64 v51; // rdi
  unsigned __int64 v52; // rdi
  __int64 v53; // rax
  int v54; // edx
  int v55; // eax
  __int64 v56; // rbx
  unsigned int v57; // eax
  unsigned __int64 v58; // r13
  __int64 v59; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v60; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v61; // [rsp+18h] [rbp-98h]
  unsigned int v62; // [rsp+20h] [rbp-90h]
  __int64 v63; // [rsp+20h] [rbp-90h]
  __int64 v64; // [rsp+28h] [rbp-88h]
  __int64 v65; // [rsp+30h] [rbp-80h]
  unsigned __int64 v67; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v68; // [rsp+48h] [rbp-68h]
  unsigned __int64 v69; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v70; // [rsp+58h] [rbp-58h]
  unsigned __int64 v71; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v72; // [rsp+68h] [rbp-48h]
  unsigned __int64 v73; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v74; // [rsp+78h] [rbp-38h]

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  v2 = *(_QWORD *)(a2 + 8);
  v3 = *(_DWORD **)a2;
  if ( v2 != *(_QWORD *)a2 )
  {
    _BitScanReverse64(&v4, 0xAAAAAAAAAAAAAAABLL * ((v2 - (__int64)v3) >> 6));
    sub_28B9AC0(v3, v2, 2LL * (int)(63 - (v4 ^ 0x3F)));
    if ( v2 - (__int64)v3 > 3072 )
    {
      v28 = (__int64 *)(v3 + 768);
      sub_28B8430((__int64)v3, (__int64)(v3 + 768));
      if ( (_DWORD *)v2 != v3 + 768 )
      {
        do
        {
          v29 = v28;
          v28 += 24;
          sub_28B5980(v29);
        }
        while ( (__int64 *)v2 != v28 );
      }
    }
    else
    {
      sub_28B8430((__int64)v3, v2);
    }
    v64 = *(_QWORD *)(a2 + 8);
    if ( v64 != *(_QWORD *)a2 )
    {
      v5 = *(_QWORD *)a2 + 24LL;
      v6 = 0;
      while ( 1 )
      {
        v65 = v5 - 24;
        if ( !v6 )
          goto LABEL_8;
        v7 = v6[1];
        if ( *(_DWORD *)(v7 - 80) != *(_DWORD *)(v5 + 88) || *(_DWORD *)(v7 - 40) != *(_DWORD *)(v5 + 128) )
          goto LABEL_8;
        v30 = *(_DWORD *)(v7 - 16);
        v72 = *(_DWORD *)(v7 - 64);
        v31 = v30 / 8;
        if ( v72 > 0x40 )
          sub_C43780((__int64)&v71, (const void **)(v7 - 72));
        else
          v71 = *(_QWORD *)(v7 - 72);
        sub_C46A40((__int64)&v71, v31);
        v32 = v72;
        v33 = v71;
        v72 = 0;
        v62 = v32;
        v74 = v32;
        v73 = v71;
        if ( v32 <= 0x40 )
        {
          if ( v71 != *(_QWORD *)(v5 + 96) )
            goto LABEL_8;
        }
        else
        {
          v34 = sub_C43C50((__int64)&v73, (const void **)(v5 + 96));
          if ( !v34 )
            goto LABEL_42;
        }
        v55 = *(_DWORD *)(v7 - 16);
        v68 = *(_DWORD *)(v7 - 24);
        v56 = v55 / 8;
        if ( v68 > 0x40 )
          sub_C43780((__int64)&v67, (const void **)(v7 - 32));
        else
          v67 = *(_QWORD *)(v7 - 32);
        sub_C46A40((__int64)&v67, v56);
        v57 = v68;
        v58 = v67;
        v68 = 0;
        v70 = v57;
        v69 = v67;
        if ( v57 <= 0x40 )
        {
          v34 = v67 == *(_QWORD *)(v5 + 136);
        }
        else
        {
          v34 = sub_C43C50((__int64)&v69, (const void **)(v5 + 136));
          if ( v58 )
          {
            j_j___libc_free_0_0(v58);
            if ( v68 > 0x40 )
            {
              if ( v67 )
                j_j___libc_free_0_0(v67);
            }
          }
        }
        if ( v62 > 0x40 )
        {
LABEL_42:
          if ( v33 )
            j_j___libc_free_0_0(v33);
        }
        if ( v72 > 0x40 && v71 )
          j_j___libc_free_0_0(v71);
        if ( v34 )
        {
          v8 = v6[1];
          if ( v8 == v6[2] )
            goto LABEL_49;
LABEL_13:
          if ( v8 )
          {
            v9 = *(_QWORD *)(v5 - 24);
            *(_QWORD *)(v8 + 8) = 0;
            v10 = v8 + 8;
            *(_QWORD *)(v8 + 16) = 1;
            v11 = v5 - 16;
            v12 = v8 + 24;
            v13 = (__int64 *)(v8 + 88);
            *(_QWORD *)v8 = v9;
            v14 = (__int64 *)(v8 + 24);
            do
            {
              if ( v14 )
                *v14 = -4096;
              ++v14;
            }
            while ( v14 != v13 );
            v15 = *(_DWORD *)(v65 + 16) & 0xFFFFFFFE;
            *(_DWORD *)(v65 + 16) = *(_DWORD *)(v8 + 16) & 0xFFFFFFFE | *(_DWORD *)(v65 + 16) & 1;
            *(_DWORD *)(v8 + 16) = v15 | *(_DWORD *)(v8 + 16) & 1;
            v16 = *(_DWORD *)(v8 + 20);
            *(_DWORD *)(v8 + 20) = *(_DWORD *)(v5 - 4);
            *(_DWORD *)(v5 - 4) = v16;
            if ( (*(_BYTE *)(v8 + 16) & 1) != 0 )
            {
              v17 = v5;
              if ( (*(_BYTE *)(v65 + 16) & 1) == 0 )
                goto LABEL_21;
              v35 = (__int64 *)(v8 + 24);
              v36 = (__int64 *)v5;
              do
              {
                v37 = *v35;
                *v35++ = *v36;
                *v36++ = v37;
              }
              while ( v13 != v35 );
            }
            else if ( (*(_BYTE *)(v65 + 16) & 1) != 0 )
            {
              v17 = v8 + 24;
              v11 = v8 + 8;
              v12 = v5;
              v10 = v5 - 16;
LABEL_21:
              *(_BYTE *)(v11 + 8) |= 1u;
              v18 = *(_QWORD *)(v11 + 16);
              v19 = 0;
              v20 = *(_DWORD *)(v11 + 24);
              do
              {
                *(_QWORD *)(v17 + v19) = *(_QWORD *)(v12 + v19);
                v19 += 8;
              }
              while ( v19 != 64 );
              *(_BYTE *)(v10 + 8) &= ~1u;
              *(_QWORD *)(v10 + 16) = v18;
              *(_DWORD *)(v10 + 24) = v20;
            }
            else
            {
              v53 = *(_QWORD *)(v8 + 24);
              *(_QWORD *)(v8 + 24) = *(_QWORD *)v5;
              v54 = *(_DWORD *)(v5 + 8);
              *(_QWORD *)v5 = v53;
              LODWORD(v53) = *(_DWORD *)(v8 + 32);
              *(_DWORD *)(v8 + 32) = v54;
              *(_DWORD *)(v5 + 8) = v53;
            }
            *(_BYTE *)(v8 + 88) = *(_BYTE *)(v5 + 64);
            *(_DWORD *)(v8 + 92) = *(_DWORD *)(v5 + 68);
            *(_QWORD *)(v8 + 96) = *(_QWORD *)(v5 + 72);
            *(_QWORD *)(v8 + 104) = *(_QWORD *)(v5 + 80);
            *(_DWORD *)(v8 + 112) = *(_DWORD *)(v5 + 88);
            *(_DWORD *)(v8 + 128) = *(_DWORD *)(v5 + 104);
            *(_QWORD *)(v8 + 120) = *(_QWORD *)(v5 + 96);
            v21 = *(_QWORD *)(v5 + 112);
            *(_DWORD *)(v5 + 104) = 0;
            *(_QWORD *)(v8 + 136) = v21;
            *(_QWORD *)(v8 + 144) = *(_QWORD *)(v5 + 120);
            *(_DWORD *)(v8 + 152) = *(_DWORD *)(v5 + 128);
            *(_DWORD *)(v8 + 168) = *(_DWORD *)(v5 + 144);
            *(_QWORD *)(v8 + 160) = *(_QWORD *)(v5 + 136);
            LODWORD(v21) = *(_DWORD *)(v5 + 152);
            *(_DWORD *)(v5 + 144) = 0;
            *(_DWORD *)(v8 + 176) = v21;
            *(_QWORD *)(v8 + 184) = *(_QWORD *)(v5 + 160);
            v8 = v6[1];
          }
          v6[1] = v8 + 192;
          goto LABEL_26;
        }
LABEL_8:
        v6 = (unsigned __int64 *)a1[1];
        if ( v6 != (unsigned __int64 *)a1[2] )
        {
          if ( v6 )
          {
            *v6 = 0;
            v6[1] = 0;
            v6[2] = 0;
            v6 = (unsigned __int64 *)a1[1];
          }
          a1[1] = (unsigned __int64)(v6 + 3);
          goto LABEL_12;
        }
        v38 = (__int64)v6 - *a1;
        v61 = *a1;
        v39 = 0xAAAAAAAAAAAAAAABLL * (v38 >> 3);
        if ( v39 == 0x555555555555555LL )
          sub_4262D8((__int64)"vector::_M_realloc_insert");
        v40 = 1;
        if ( v39 )
          v40 = 0xAAAAAAAAAAAAAAABLL * (v38 >> 3);
        v41 = __CFADD__(v40, v39);
        v42 = v40 - 0x5555555555555555LL * (v38 >> 3);
        if ( v41 )
        {
          v43 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_62:
          v63 = sub_22077B0(v43);
          v44 = v63 + 24;
          v60 = v63 + v43;
          goto LABEL_63;
        }
        if ( v42 )
        {
          if ( v42 > 0x555555555555555LL )
            v42 = 0x555555555555555LL;
          v43 = 24 * v42;
          goto LABEL_62;
        }
        v60 = 0;
        v44 = 24;
        v63 = 0;
LABEL_63:
        v45 = (_QWORD *)(v63 + v38);
        if ( v45 )
        {
          *v45 = 0;
          v45[1] = 0;
          v45[2] = 0;
        }
        v46 = (unsigned __int64 *)v61;
        if ( v6 != (unsigned __int64 *)v61 )
        {
          v59 = v5;
          for ( i = (_QWORD *)v63; ; i = (_QWORD *)v48 )
          {
            v49 = v46[1];
            v50 = *v46;
            if ( i )
              break;
            if ( v50 != v49 )
            {
              do
              {
                if ( *(_DWORD *)(v50 + 168) > 0x40u )
                {
                  v51 = *(_QWORD *)(v50 + 160);
                  if ( v51 )
                    j_j___libc_free_0_0(v51);
                }
                if ( *(_DWORD *)(v50 + 128) > 0x40u )
                {
                  v52 = *(_QWORD *)(v50 + 120);
                  if ( v52 )
                    j_j___libc_free_0_0(v52);
                }
                if ( (*(_BYTE *)(v50 + 16) & 1) == 0 )
                  sub_C7D6A0(*(_QWORD *)(v50 + 24), 8LL * *(unsigned int *)(v50 + 32), 8);
                v50 += 192LL;
              }
              while ( v49 != v50 );
              v50 = *v46;
            }
            if ( !v50 )
              goto LABEL_68;
            v46 += 3;
            j_j___libc_free_0(v50);
            v48 = 24;
            if ( v6 == v46 )
            {
LABEL_84:
              v5 = v59;
              v44 = (__int64)(i + 6);
              v6 = (unsigned __int64 *)v48;
              goto LABEL_85;
            }
LABEL_69:
            ;
          }
          *i = v50;
          i[1] = v49;
          i[2] = v46[2];
          v46[2] = 0;
          v46[1] = 0;
          *v46 = 0;
LABEL_68:
          v46 += 3;
          v48 = (__int64)(i + 3);
          if ( v6 == v46 )
            goto LABEL_84;
          goto LABEL_69;
        }
        v6 = (unsigned __int64 *)v63;
LABEL_85:
        if ( v61 )
          j_j___libc_free_0(v61);
        *a1 = v63;
        a1[1] = v44;
        a1[2] = v60;
LABEL_12:
        v8 = v6[1];
        if ( v8 != v6[2] )
          goto LABEL_13;
LABEL_49:
        sub_28B5090(v6, v8, (__int64 *)v65);
LABEL_26:
        if ( v64 == v5 + 168 )
          break;
        v5 += 192;
      }
    }
  }
  v22 = (char *)a1[1];
  v23 = (char *)*a1;
  if ( v22 != (char *)*a1 )
  {
    _BitScanReverse64(&v24, 0xAAAAAAAAAAAAAAABLL * ((v22 - v23) >> 3));
    sub_28B93F0((_QWORD *)*a1, (char *)a1[1], 2LL * (int)(63 - (v24 ^ 0x3F)));
    if ( v22 - v23 <= 384 )
    {
      sub_28B5F90(v23, v22);
    }
    else
    {
      v25 = v23 + 384;
      sub_28B5F90(v23, v23 + 384);
      if ( v22 != v23 + 384 )
      {
        do
        {
          v26 = v25;
          v25 += 24;
          sub_28B5D30(v26);
        }
        while ( v22 != v25 );
      }
    }
  }
  return a1;
}
