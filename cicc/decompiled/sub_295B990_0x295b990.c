// Function: sub_295B990
// Address: 0x295b990
//
__int64 *__fastcall sub_295B990(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rsi
  _BYTE *v12; // rdi
  unsigned int v13; // eax
  __int64 *v14; // rdx
  unsigned __int64 v15; // rcx
  __int64 *v16; // rbx
  __int64 *v17; // r8
  __int64 *v18; // r15
  _BYTE *v19; // rax
  __int64 v20; // r14
  __int64 v21; // r14
  __int64 v22; // r8
  unsigned __int64 v23; // r9
  __int64 v24; // rax
  __int64 v26; // rdi
  __int64 v27; // r8
  __int64 v28; // r9
  _BYTE *v29; // rdi
  __int64 *v30; // rax
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  __int64 v33; // rdi
  __int64 v34; // rsi
  _BYTE *v35; // rdi
  __int64 v36; // rax
  __int64 v37; // r9
  __int64 v38; // rdx
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // r9
  unsigned __int64 v43; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v44; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v45; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v46; // [rsp+20h] [rbp-D0h]
  bool v47; // [rsp+2Eh] [rbp-C2h]
  bool v48; // [rsp+2Fh] [rbp-C1h]
  __int64 *v49; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v50; // [rsp+38h] [rbp-B8h]
  _QWORD v51[4]; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v52; // [rsp+60h] [rbp-90h] BYREF
  __int64 *v53; // [rsp+68h] [rbp-88h]
  __int64 v54; // [rsp+70h] [rbp-80h]
  int v55; // [rsp+78h] [rbp-78h]
  char v56; // [rsp+7Ch] [rbp-74h]
  char v57; // [rsp+80h] [rbp-70h] BYREF

  *a1 = 0;
  v6 = *(_QWORD *)(a3 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 <= 1 )
    v6 = **(_QWORD **)(v6 + 16);
  v48 = sub_BCAC40(v6, 1);
  if ( v48 )
  {
    v7 = *(_QWORD *)(a3 + 8);
    if ( *(_BYTE *)a3 == 57 )
      goto LABEL_10;
    v48 = 0;
    if ( *(_BYTE *)a3 != 86 || *(_QWORD *)(*(_QWORD *)(a3 - 96) + 8LL) != v7 || **(_BYTE **)(a3 - 32) > 0x15u )
      goto LABEL_10;
    v48 = sub_AC30F0(*(_QWORD *)(a3 - 32));
  }
  v7 = *(_QWORD *)(a3 + 8);
LABEL_10:
  if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
    v7 = **(_QWORD **)(v7 + 16);
  v47 = sub_BCAC40(v7, 1);
  if ( v47 && *(_BYTE *)a3 != 58 )
  {
    v47 = 0;
    if ( *(_BYTE *)a3 == 86 )
    {
      v11 = *(_QWORD *)(a3 + 8);
      if ( *(_QWORD *)(*(_QWORD *)(a3 - 96) + 8LL) == v11 )
      {
        v12 = *(_BYTE **)(a3 - 64);
        if ( *v12 <= 0x15u )
          v47 = sub_AD7A80(v12, v11, v8, v9, v10);
      }
    }
  }
  v52 = 0;
  v49 = v51;
  v53 = (__int64 *)&v57;
  v54 = 8;
  v55 = 0;
  v56 = 1;
  v51[0] = a3;
  v50 = 0x400000001LL;
  sub_AE6EC0((__int64)&v52, a3);
  v13 = 1;
  do
  {
    v14 = v49;
    v15 = v13--;
    v16 = (__int64 *)v49[v15 - 1];
    LODWORD(v50) = v13;
    if ( (*((_BYTE *)v16 + 7) & 0x40) != 0 )
    {
      v17 = (__int64 *)*(v16 - 1);
      v16 = &v17[4 * (*((_DWORD *)v16 + 1) & 0x7FFFFFF)];
    }
    else
    {
      v14 = (__int64 *)(32LL * (*((_DWORD *)v16 + 1) & 0x7FFFFFF));
      v17 = (__int64 *)((char *)v16 - (char *)v14);
    }
    if ( v17 != v16 )
    {
      v18 = v17;
      while ( 1 )
      {
        while ( 1 )
        {
          v21 = *v18;
          if ( *(_BYTE *)*v18 <= 0x15u )
            goto LABEL_26;
          if ( !(unsigned __int8)sub_D48480(a2, *v18, (__int64)v14, v15) )
            break;
          v23 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v23 )
          {
            *a1 = v21 & 0xFFFFFFFFFFFFFFFBLL;
            goto LABEL_26;
          }
          if ( (*a1 & 4) == 0 )
          {
            v44 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
            v36 = sub_22077B0(0x30u);
            v37 = v44;
            if ( v36 )
            {
              *(_QWORD *)v36 = v36 + 16;
              *(_QWORD *)(v36 + 8) = 0x400000000LL;
            }
            v38 = v36;
            v39 = v36 & 0xFFFFFFFFFFFFFFF8LL;
            v40 = *(unsigned int *)(v39 + 12);
            *a1 = v38 | 4;
            v41 = *(unsigned int *)(v39 + 8);
            if ( v41 + 1 > v40 )
            {
              v43 = v44;
              v46 = v39;
              sub_C8D5F0(v39, (const void *)(v39 + 16), v41 + 1, 8u, v22, v37);
              v39 = v46;
              v37 = v43;
              v41 = *(unsigned int *)(v46 + 8);
            }
            *(_QWORD *)(*(_QWORD *)v39 + 8 * v41) = v37;
            v42 = *a1;
            ++*(_DWORD *)(v39 + 8);
            v23 = v42 & 0xFFFFFFFFFFFFFFF8LL;
          }
          v24 = *(unsigned int *)(v23 + 8);
          v15 = *(unsigned int *)(v23 + 12);
          if ( v24 + 1 > v15 )
          {
            v45 = v23;
            sub_C8D5F0(v23, (const void *)(v23 + 16), v24 + 1, 8u, v22, v23);
            v23 = v45;
            v24 = *(unsigned int *)(v45 + 8);
          }
          v14 = *(__int64 **)v23;
          v18 += 4;
          *(_QWORD *)(*(_QWORD *)v23 + 8 * v24) = v21;
          ++*(_DWORD *)(v23 + 8);
          if ( v16 == v18 )
          {
LABEL_34:
            v13 = v50;
            goto LABEL_35;
          }
        }
        v19 = sub_2958930((_BYTE *)v21);
        v20 = (__int64)v19;
        if ( *v19 > 0x1Cu )
          break;
LABEL_26:
        v18 += 4;
        if ( v16 == v18 )
          goto LABEL_34;
      }
      if ( v48 )
      {
        v26 = *((_QWORD *)v19 + 1);
        if ( (unsigned int)*(unsigned __int8 *)(v26 + 8) - 17 <= 1 )
          v26 = **(_QWORD **)(v26 + 16);
        if ( sub_BCAC40(v26, 1) )
        {
          if ( *(_BYTE *)v20 == 57 )
            goto LABEL_51;
          if ( *(_BYTE *)v20 == 86 && *(_QWORD *)(*(_QWORD *)(v20 - 96) + 8LL) == *(_QWORD *)(v20 + 8) )
          {
            v29 = *(_BYTE **)(v20 - 32);
            if ( *v29 <= 0x15u && sub_AC30F0((__int64)v29) )
              goto LABEL_51;
          }
        }
      }
      if ( !v47 )
        goto LABEL_26;
      v33 = *(_QWORD *)(v20 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v33 + 8) - 17 <= 1 )
        v33 = **(_QWORD **)(v33 + 16);
      if ( !sub_BCAC40(v33, 1) )
        goto LABEL_26;
      if ( *(_BYTE *)v20 == 58 )
      {
LABEL_51:
        if ( !v56 )
        {
LABEL_69:
          sub_C8CC70((__int64)&v52, v20, (__int64)v14, v15, v27, v28);
          if ( (_BYTE)v14 )
            goto LABEL_57;
          goto LABEL_26;
        }
      }
      else
      {
        if ( *(_BYTE *)v20 != 86 )
          goto LABEL_26;
        v34 = *(_QWORD *)(v20 + 8);
        if ( *(_QWORD *)(*(_QWORD *)(v20 - 96) + 8LL) != v34 )
          goto LABEL_26;
        v35 = *(_BYTE **)(v20 - 64);
        if ( *v35 > 0x15u || !sub_AD7A80(v35, v34, (__int64)v14, v15, v27) )
          goto LABEL_26;
        if ( !v56 )
          goto LABEL_69;
      }
      v30 = v53;
      v15 = HIDWORD(v54);
      v14 = &v53[HIDWORD(v54)];
      if ( v53 != v14 )
      {
        while ( v20 != *v30 )
        {
          if ( v14 == ++v30 )
            goto LABEL_55;
        }
        goto LABEL_26;
      }
LABEL_55:
      if ( HIDWORD(v54) < (unsigned int)v54 )
      {
        ++HIDWORD(v54);
        *v14 = v20;
        ++v52;
LABEL_57:
        v31 = (unsigned int)v50;
        v15 = HIDWORD(v50);
        v32 = (unsigned int)v50 + 1LL;
        if ( v32 > HIDWORD(v50) )
        {
          sub_C8D5F0((__int64)&v49, v51, v32, 8u, v27, v28);
          v31 = (unsigned int)v50;
        }
        v14 = v49;
        v49[v31] = v20;
        LODWORD(v50) = v50 + 1;
        goto LABEL_26;
      }
      goto LABEL_69;
    }
LABEL_35:
    ;
  }
  while ( v13 );
  if ( !v56 )
    _libc_free((unsigned __int64)v53);
  if ( v49 != v51 )
    _libc_free((unsigned __int64)v49);
  return a1;
}
