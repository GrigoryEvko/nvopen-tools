// Function: sub_7A44E0
// Address: 0x7a44e0
//
__int64 __fastcall sub_7A44E0(__int64 a1, __int64 a2, __int64 a3, char **a4, __int64 a5, __int64 a6, __m128i a7)
{
  char *v10; // r14
  char v11; // si
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 i; // rax
  __int64 j; // rax
  FILE *v16; // rbx
  unsigned int v17; // r15d
  __int64 v18; // rsi
  _QWORD *v20; // rdx
  FILE *v21; // rsi
  unsigned int v22; // edi
  char v23; // al
  __int64 v24; // rdi
  char v25; // si
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rdx
  __int64 v29; // rcx
  const __m128i *v30; // r15
  __int8 v31; // al
  __int64 v32; // r13
  char v33; // al
  _QWORD *v34; // rax
  __int64 *v35; // rcx
  _QWORD *v36; // r13
  char v37; // al
  _QWORD *v38; // rax
  _QWORD *v39; // rax
  char v40; // dl
  __int64 v41; // rax
  __int64 *v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // r13
  char v48; // dl
  __int64 **v49; // rax
  __int64 *v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rax
  char v53; // dl
  __int64 v54; // r15
  char v55; // dl
  __int64 **v56; // rax
  __int64 *v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // r15
  _BYTE *v62; // rax
  __int64 v63; // r12
  __int64 v64; // [rsp-20h] [rbp-D0h]
  __int64 *v65; // [rsp+8h] [rbp-A8h]
  __int64 *v66; // [rsp+8h] [rbp-A8h]
  __int64 *v67; // [rsp+8h] [rbp-A8h]
  __int64 v68; // [rsp+10h] [rbp-A0h]
  _QWORD *v71; // [rsp+28h] [rbp-88h]
  const __m128i *v72; // [rsp+28h] [rbp-88h]
  __int64 v73; // [rsp+38h] [rbp-78h] BYREF
  const __m128i *v74; // [rsp+40h] [rbp-70h] BYREF
  __int64 v75; // [rsp+48h] [rbp-68h]
  __int64 v76; // [rsp+50h] [rbp-60h]
  __m128i v77; // [rsp+60h] [rbp-50h] BYREF
  __int64 v78; // [rsp+70h] [rbp-40h]

  v10 = *a4;
  v11 = **a4;
  v12 = *((_QWORD *)*a4 + 1);
  if ( v11 == 48 )
  {
    v23 = *(_BYTE *)(v12 + 8);
    if ( v23 == 1 )
    {
      v12 = *(_QWORD *)(v12 + 32);
      v11 = 2;
    }
    else if ( v23 == 2 )
    {
      v12 = *(_QWORD *)(v12 + 32);
      v11 = 59;
    }
    else
    {
      if ( v23 )
        goto LABEL_19;
      v12 = *(_QWORD *)(v12 + 32);
      v11 = 6;
    }
  }
  v71 = (_QWORD *)sub_72A270(v12, v11);
  v76 = 0;
  v75 = 0;
  v74 = (const __m128i *)sub_823970(0);
  v13 = (__int64)v74;
  for ( i = *(_QWORD *)(a2 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( (*(_BYTE *)(a1 + 132) & 1) == 0 )
  {
    v18 = 0;
    v17 = 0;
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_6855B0(0xAA1u, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      v13 = (__int64)v74;
      v18 = 24 * v75;
    }
    goto LABEL_13;
  }
  for ( j = *(_QWORD *)(***(_QWORD ***)(i + 168) + 8LL); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  if ( *v10 == 48 )
  {
    v24 = *((_QWORD *)v10 + 1);
    v25 = *(_BYTE *)(v24 + 8);
    switch ( v25 )
    {
      case 1:
        *v10 = 2;
        *((_QWORD *)v10 + 1) = *(_QWORD *)(v24 + 32);
        goto LABEL_8;
      case 2:
        *v10 = 59;
        *((_QWORD *)v10 + 1) = *(_QWORD *)(v24 + 32);
        goto LABEL_8;
      case 0:
        *v10 = 6;
        *((_QWORD *)v10 + 1) = *(_QWORD *)(v24 + 32);
        goto LABEL_8;
    }
LABEL_19:
    sub_721090();
  }
LABEL_8:
  v16 = (FILE *)(a3 + 28);
  if ( !v71 || !*v71 || *v10 != 59 )
  {
    if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
    {
LABEL_12:
      v13 = (__int64)v74;
      v17 = 0;
      v18 = 24 * v75;
      goto LABEL_13;
    }
    v20 = (_QWORD *)(a1 + 96);
    v21 = v16;
    v22 = 3375;
LABEL_15:
    sub_6855B0(v22, v21, v20);
    sub_770D30(a1);
    goto LABEL_12;
  }
  v68 = **(_QWORD **)(j + 168);
  if ( v68 )
  {
    if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
      goto LABEL_12;
    v20 = (_QWORD *)(a1 + 96);
    v21 = (FILE *)(a1 + 112);
    v22 = 3367;
    goto LABEL_15;
  }
  v17 = sub_7A4160(a1, *(_QWORD *)(j + 160), (__int64 *)&v74, v16);
  if ( !v17 )
  {
    v13 = (__int64)v74;
    v18 = 24 * v75;
    goto LABEL_13;
  }
  v28 = (__int64)v74;
  v73 = 0;
  v29 = (__int64)&v73;
  v30 = v74;
  v72 = (const __m128i *)((char *)v74 + 24 * v76);
  if ( v72 != v74 )
  {
    do
    {
      a7 = _mm_loadu_si128(v30);
      v77 = a7;
      v78 = v30[1].m128i_i64[0];
      v31 = v30->m128i_i8[0];
      v32 = v30->m128i_i64[1];
      if ( v30->m128i_i8[0] == 48 )
      {
        v33 = *(_BYTE *)(v32 + 8);
        if ( v33 == 1 )
        {
          v32 = *(_QWORD *)(v32 + 32);
LABEL_54:
          v67 = (__int64 *)v29;
          v39 = sub_725090(1u);
          v35 = v67;
          *v67 = (__int64)v39;
          v39[4] = v32;
          goto LABEL_45;
        }
        if ( v33 == 2 )
        {
          v32 = *(_QWORD *)(v32 + 32);
LABEL_52:
          v66 = (__int64 *)v29;
          v38 = sub_725090(2u);
          v35 = v66;
          *v66 = (__int64)v38;
          v38[4] = v32;
          goto LABEL_45;
        }
        if ( v33 )
          goto LABEL_19;
        v32 = *(_QWORD *)(v32 + 32);
      }
      else if ( v31 != 6 )
      {
        if ( v31 != 59 )
        {
          if ( v31 != 2 )
          {
            LODWORD(v64) = v77.m128i_i32[0];
            goto LABEL_37;
          }
          goto LABEL_54;
        }
        goto LABEL_52;
      }
      v65 = (__int64 *)v29;
      v34 = sub_725090(0);
      v35 = v65;
      *v65 = (__int64)v34;
      v34[4] = v32;
LABEL_45:
      v29 = *v35;
      v30 = (const __m128i *)((char *)v30 + 24);
    }
    while ( v72 != v30 );
  }
  v36 = (_QWORD *)*((_QWORD *)v10 + 1);
  v37 = *((_BYTE *)v36 + 120);
  switch ( v37 )
  {
    case 1:
      v54 = *v36;
      v55 = *(_BYTE *)(*v36 + 80LL);
      v56 = *(__int64 ***)(*v36 + 88LL);
      if ( v55 == 20 )
      {
        v58 = *v56[41];
      }
      else
      {
        if ( v55 == 21 )
          v57 = v56[29];
        else
          v57 = v56[4];
        v58 = *v57;
      }
      if ( (unsigned int)sub_8AEF00(v54, &v73, v58, v29) )
      {
        v59 = sub_8A0370(v54, (unsigned int)&v73, 0, 0, 0, 0, 1);
        if ( v59 )
        {
          *(_BYTE *)a5 = 6;
          v60 = *(_QWORD *)(v59 + 88);
          *(_DWORD *)(a5 + 16) = 0;
          *(_QWORD *)(a5 + 8) = v60;
          goto LABEL_66;
        }
      }
      break;
    case 3:
      v47 = *v36;
      v48 = *(_BYTE *)(v47 + 80);
      v49 = *(__int64 ***)(v47 + 88);
      if ( v48 == 20 )
      {
        v51 = *v49[41];
      }
      else
      {
        if ( v48 == 21 )
          v50 = v49[29];
        else
          v50 = v49[4];
        v51 = *v50;
      }
      if ( (unsigned int)sub_8AEF00(v47, &v73, v51, v29) )
      {
        v52 = sub_8C0230(v47, &v73, 1, 0, 0);
        if ( v52 )
        {
          v29 = *(unsigned __int8 *)(v52 + 80);
          v28 = (unsigned int)(v29 - 7);
          LOBYTE(v28) = (v29 - 7) & 0xFD;
          if ( !(_BYTE)v28 )
          {
            *(_BYTE *)a5 = 7;
            v53 = *(_BYTE *)(v52 + 80);
            if ( v53 == 9 || v53 == 7 )
            {
              v68 = *(_QWORD *)(v52 + 88);
            }
            else if ( v53 == 21 )
            {
              v68 = *(_QWORD *)(*(_QWORD *)(v52 + 88) + 192LL);
            }
            *(_DWORD *)(a5 + 16) = 0;
            *(_QWORD *)(a5 + 8) = v68;
            goto LABEL_66;
          }
        }
      }
      break;
    case 9:
      v61 = **(_QWORD **)(*(_QWORD *)(*v36 + 88LL) + 32LL);
      v62 = sub_724D80(1);
      v77 = 0u;
      v63 = (__int64)v62;
      LODWORD(v61) = sub_6F1C10(v36[24], v73, v61, &v77, 0, 0, 0, 0);
      sub_67E3D0(&v77);
      sub_72C470(v61, v63);
      *(_BYTE *)a5 = 2;
      *(_QWORD *)(a5 + 8) = v63;
      *(_DWORD *)(a5 + 16) = 0;
      goto LABEL_66;
    case 2:
      v40 = *(_BYTE *)(*v36 + 80LL);
      v41 = *(_QWORD *)(*v36 + 88LL);
      if ( v40 == 20 )
      {
        v43 = **(_QWORD **)(v41 + 328);
      }
      else
      {
        v42 = v40 == 21 ? *(__int64 **)(v41 + 232) : *(__int64 **)(v41 + 32);
        v43 = *v42;
      }
      if ( (unsigned int)sub_8AEF00(*v36, &v73, v43, v29) )
      {
        v44 = sub_8B74F0(*v36, &v73, 0, v16);
        if ( v44 )
        {
          v29 = *(unsigned __int8 *)(v44 + 80);
          v28 = (unsigned int)(v29 - 10);
          if ( (unsigned __int8)(v29 - 10) <= 1u )
          {
            *(_BYTE *)a5 = 11;
            v45 = *(_QWORD *)(v44 + 88);
            *(_DWORD *)(a5 + 16) = 0;
            *(_QWORD *)(a5 + 8) = v45;
LABEL_66:
            v17 = 1;
            v13 = (__int64)v74;
            v46 = -(((unsigned int)(a5 - a6) >> 3) + 10);
            *(_BYTE *)(a6 + v46) |= 1 << ((a5 - a6) & 7);
            v18 = 24 * v75;
            goto LABEL_13;
          }
        }
      }
      break;
  }
  v64 = *(_QWORD *)v10;
LABEL_37:
  v17 = 0;
  sub_770ED0(v16, a1, v28, v29, v26, v27, a7, v64);
  v13 = (__int64)v74;
  v18 = 24 * v75;
LABEL_13:
  sub_823A00(v13, v18);
  return v17;
}
