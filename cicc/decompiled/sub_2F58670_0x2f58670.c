// Function: sub_2F58670
// Address: 0x2f58670
//
void __fastcall sub_2F58670(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  int v6; // r12d
  int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned int v10; // r15d
  __int64 v11; // r12
  unsigned int v12; // eax
  unsigned int v13; // edx
  __int64 v14; // rbx
  int v15; // r13d
  __int64 v16; // r10
  __int64 v17; // rax
  unsigned __int64 v18; // rcx
  __int64 v19; // r11
  _QWORD *v20; // rbx
  __int64 v21; // rax
  int v22; // edx
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rsi
  __int64 v26; // rdx
  unsigned int *v27; // rdi
  bool v28; // cf
  unsigned int v29; // esi
  __int64 v30; // rsi
  __int64 *v31; // r13
  __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // r11
  unsigned __int64 v35; // r13
  _QWORD *v36; // rdx
  _QWORD *v37; // rdi
  unsigned int *v38; // rbx
  unsigned int *i; // r12
  unsigned int *v40; // rdx
  __int64 v41; // rdx
  _QWORD *v42; // [rsp+50h] [rbp-120h]
  unsigned int v43; // [rsp+50h] [rbp-120h]
  __int64 v44; // [rsp+50h] [rbp-120h]
  __int64 v45; // [rsp+58h] [rbp-118h]
  __int64 v46; // [rsp+58h] [rbp-118h]
  __int64 v47; // [rsp+58h] [rbp-118h]
  __int64 v48; // [rsp+58h] [rbp-118h]
  _DWORD *v49; // [rsp+60h] [rbp-110h] BYREF
  __int64 v50; // [rsp+68h] [rbp-108h]
  _DWORD v51[4]; // [rsp+70h] [rbp-100h] BYREF
  _BYTE v52[32]; // [rsp+80h] [rbp-F0h] BYREF
  _DWORD *v53; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 v54; // [rsp+A8h] [rbp-C8h]
  _DWORD v55[6]; // [rsp+B0h] [rbp-C0h] BYREF
  int v56; // [rsp+C8h] [rbp-A8h] BYREF
  unsigned __int64 v57; // [rsp+D0h] [rbp-A0h]
  int *v58; // [rsp+D8h] [rbp-98h]
  int *v59; // [rsp+E0h] [rbp-90h]
  __int64 v60; // [rsp+E8h] [rbp-88h]
  unsigned int *v61; // [rsp+F0h] [rbp-80h] BYREF
  __int64 v62; // [rsp+F8h] [rbp-78h]
  _BYTE v63[112]; // [rsp+100h] [rbp-70h] BYREF

  v53 = v55;
  v6 = *(_DWORD *)(a2 + 112);
  v54 = 0x400000000LL;
  v7 = v6;
  v62 = 0x400000000LL;
  v8 = *(_QWORD *)(a1 + 24);
  v58 = &v56;
  v59 = &v56;
  v56 = 0;
  v57 = 0;
  v60 = 0;
  v9 = *(_QWORD *)(v8 + 32);
  v49 = v51;
  v50 = 0x200000000LL;
  v61 = (unsigned int *)v63;
  v10 = *(_DWORD *)(v9 + 4LL * (v6 & 0x7FFFFFFF));
  v55[0] = v6;
  v51[0] = v6;
  v11 = a1;
  v12 = 1;
  LODWORD(v54) = 1;
  while ( 1 )
  {
    LODWORD(v50) = --v12;
    if ( (unsigned int)(v7 - 1) <= 0x3FFFFFFE )
      goto LABEL_26;
    v13 = v7 & 0x7FFFFFFF;
    v14 = v7 & 0x7FFFFFFF;
    v15 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v11 + 24) + 32LL) + 4 * v14);
    if ( !v15 )
      goto LABEL_26;
    v16 = *(_QWORD *)(v11 + 32);
    v17 = 8 * v14;
    v18 = *(unsigned int *)(v16 + 160);
    if ( v13 >= (unsigned int)v18 || (v19 = *(_QWORD *)(*(_QWORD *)(v16 + 152) + 8 * v14)) == 0 )
    {
      v29 = v13 + 1;
      if ( (unsigned int)v18 < v13 + 1 && v29 != v18 )
      {
        if ( v29 >= v18 )
        {
          v34 = *(_QWORD *)(v16 + 168);
          v35 = v29 - v18;
          if ( v29 > (unsigned __int64)*(unsigned int *)(v16 + 164) )
          {
            v44 = *(_QWORD *)(v16 + 168);
            v48 = *(_QWORD *)(v11 + 32);
            sub_C8D5F0(v16 + 152, (const void *)(v16 + 168), v29, 8u, a5, a6);
            v16 = v48;
            v17 = 8 * v14;
            v34 = v44;
            v18 = *(unsigned int *)(v48 + 160);
          }
          v30 = *(_QWORD *)(v16 + 152);
          v36 = (_QWORD *)(v30 + 8 * v18);
          v37 = &v36[v35];
          if ( v36 != v37 )
          {
            do
              *v36++ = v34;
            while ( v37 != v36 );
            LODWORD(v18) = *(_DWORD *)(v16 + 160);
            v30 = *(_QWORD *)(v16 + 152);
          }
          *(_DWORD *)(v16 + 160) = v35 + v18;
LABEL_30:
          v31 = (__int64 *)(v30 + v17);
          v42 = (_QWORD *)v16;
          v32 = sub_2E10F30(v7);
          *v31 = v32;
          v46 = v32;
          sub_2E11E80(v42, v32);
          v19 = v46;
          v15 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v11 + 24) + 32LL) + 4 * v14);
          if ( v10 == v15 )
          {
LABEL_31:
            LODWORD(v62) = 0;
            sub_2F54E60(v11, v7, (__int64)&v61);
LABEL_49:
            v38 = &v61[4 * (unsigned int)v62];
            if ( v38 == v61 )
              goto LABEL_25;
            v47 = v11;
            for ( i = v61 + 2; ; i = v40 )
            {
              sub_2F58490((__int64)v52, (__int64)&v53, i, v33, a5);
              if ( v52[16] )
              {
                v41 = (unsigned int)v50;
                a6 = *i;
                a5 = (unsigned int)v50 + 1LL;
                if ( a5 > HIDWORD(v50) )
                {
                  v43 = *i;
                  sub_C8D5F0((__int64)&v49, v51, (unsigned int)v50 + 1LL, 4u, a5, a6);
                  v41 = (unsigned int)v50;
                  a6 = v43;
                }
                v33 = (__int64)v49;
                v49[v41] = a6;
                v40 = i + 4;
                LODWORD(v50) = v50 + 1;
                if ( v38 == i + 2 )
                {
LABEL_57:
                  v11 = v47;
                  goto LABEL_25;
                }
              }
              else
              {
                v40 = i + 4;
                if ( v38 == i + 2 )
                  goto LABEL_57;
              }
            }
          }
          goto LABEL_7;
        }
        *(_DWORD *)(v16 + 160) = v29;
      }
      v30 = *(_QWORD *)(v16 + 152);
      goto LABEL_30;
    }
    if ( v10 == v15 )
      goto LABEL_31;
LABEL_7:
    v20 = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(v11 + 16) + 56LL) + 16 * v14);
    if ( v10 - 1 <= 0x3FFFFFFE )
    {
      v21 = *(_QWORD *)(*v20 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v10 >> 3 < *(unsigned __int16 *)(v21 + 22) )
      {
        v22 = *(unsigned __int8 *)(*(_QWORD *)(v21 + 8) + (v10 >> 3));
        if ( _bittest(&v22, v10 & 7) )
        {
          v45 = v19;
          if ( !(unsigned int)sub_2E21680(*(_QWORD **)(v11 + 40), v19, v10) )
          {
            LODWORD(v62) = 0;
            sub_2F54E60(v11, v7, (__int64)&v61);
            v24 = (unsigned __int64)v61;
            v25 = 0;
            v26 = (__int64)v61;
            v27 = &v61[4 * (unsigned int)v62];
            if ( v61 == v27 )
              goto LABEL_48;
            while ( 1 )
            {
              if ( *(_DWORD *)(v26 + 12) != v15 )
              {
                v28 = __CFADD__(*(_QWORD *)v26, v25);
                v25 += *(_QWORD *)v26;
                if ( v28 )
                  v25 = -1;
              }
              if ( v27 == (unsigned int *)(v26 + 16) )
                break;
              v26 += 16;
            }
            v23 = 0;
            a5 = -1;
            while ( 1 )
            {
              if ( v10 != *(_DWORD *)(v24 + 12) )
              {
                v28 = __CFADD__(*(_QWORD *)v24, v23);
                v23 += *(_QWORD *)v24;
                if ( v28 )
                  v23 = -1;
              }
              if ( v26 == v24 )
                break;
              v24 += 16LL;
            }
            if ( v23 <= v25 )
            {
LABEL_48:
              sub_2E21040(*(_QWORD **)(v11 + 40), v45, v26, v23, -1);
              sub_2E20EE0(*(_QWORD **)(v11 + 40), v45, v10);
              goto LABEL_49;
            }
          }
        }
      }
    }
LABEL_25:
    v12 = v50;
LABEL_26:
    if ( !v12 )
      break;
    v7 = v49[v12 - 1];
  }
  if ( v61 != (unsigned int *)v63 )
    _libc_free((unsigned __int64)v61);
  if ( v49 != v51 )
    _libc_free((unsigned __int64)v49);
  sub_2F4E180(v57);
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
}
