// Function: sub_2D08E40
// Address: 0x2d08e40
//
void __fastcall sub_2D08E40(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r14
  __int64 v9; // rsi
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  _QWORD **v16; // rdx
  _QWORD **v17; // rdi
  _QWORD *v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 *v22; // rdx
  unsigned int v23; // eax
  __int64 v24; // r14
  __int64 v25; // r8
  __int64 v26; // r13
  __int64 i; // r15
  __int64 v28; // rbx
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rax
  __int64 *v36; // rcx
  __int64 v37; // rax
  _BYTE *v38; // rsi
  int v39; // edx
  __int64 v40; // rdi
  int v41; // esi
  unsigned int v42; // edx
  __int64 v43; // rcx
  __int64 *v44; // rax
  __int64 v45; // rax
  __int64 *v46; // rax
  int v47; // edi
  unsigned int v48; // esi
  _QWORD *v49; // rax
  int v50; // edx
  __int64 v51; // rdx
  int v52; // r8d
  __int64 v53; // [rsp+20h] [rbp-1F0h] BYREF
  __int64 v54; // [rsp+28h] [rbp-1E8h] BYREF
  _QWORD *v55; // [rsp+30h] [rbp-1E0h] BYREF
  _QWORD *v56; // [rsp+38h] [rbp-1D8h] BYREF
  unsigned __int64 v57; // [rsp+40h] [rbp-1D0h] BYREF
  __int64 *v58; // [rsp+48h] [rbp-1C8h]
  __int64 *v59; // [rsp+50h] [rbp-1C0h]
  __int64 v60; // [rsp+58h] [rbp-1B8h]
  _BYTE *v61; // [rsp+60h] [rbp-1B0h] BYREF
  __int64 v62; // [rsp+68h] [rbp-1A8h]
  _BYTE v63[64]; // [rsp+70h] [rbp-1A0h] BYREF
  __int64 *v64; // [rsp+B0h] [rbp-160h] BYREF
  __int64 v65; // [rsp+B8h] [rbp-158h]
  _QWORD v66[16]; // [rsp+C0h] [rbp-150h] BYREF
  __int64 v67; // [rsp+140h] [rbp-D0h] BYREF
  _BYTE *v68; // [rsp+148h] [rbp-C8h]
  __int64 v69; // [rsp+150h] [rbp-C0h]
  int v70; // [rsp+158h] [rbp-B8h]
  char v71; // [rsp+15Ch] [rbp-B4h]
  _BYTE v72[176]; // [rsp+160h] [rbp-B0h] BYREF

  v6 = *(_QWORD *)(a2 + 16);
  v61 = v63;
  v62 = 0x800000000LL;
  if ( !v6 )
  {
    v67 = 0;
    v68 = v72;
    v69 = 16;
    v70 = 0;
    v71 = 1;
    goto LABEL_28;
  }
  do
  {
    v9 = sub_2D05370(a1, v6, 0);
    if ( !v9 )
      goto LABEL_11;
    if ( *(_BYTE *)(a4 + 28) )
    {
      v12 = *(_QWORD **)(a4 + 8);
      v13 = &v12[*(unsigned int *)(a4 + 20)];
      if ( v12 == v13 )
        goto LABEL_11;
      while ( v9 != *v12 )
      {
        if ( v13 == ++v12 )
          goto LABEL_11;
      }
    }
    else if ( !sub_C8CA60(a4, v9) )
    {
      goto LABEL_11;
    }
    v14 = (unsigned int)v62;
    v15 = (unsigned int)v62 + 1LL;
    if ( v15 > HIDWORD(v62) )
    {
      sub_C8D5F0((__int64)&v61, v63, v15, 8u, v10, v11);
      v14 = (unsigned int)v62;
    }
    *(_QWORD *)&v61[8 * v14] = v6;
    LODWORD(v62) = v62 + 1;
LABEL_11:
    v6 = *(_QWORD *)(v6 + 8);
  }
  while ( v6 );
  v16 = (_QWORD **)v61;
  v17 = (_QWORD **)&v61[8 * (unsigned int)v62];
  if ( v17 != (_QWORD **)v61 )
  {
    do
    {
      v18 = *v16;
      if ( **v16 )
      {
        v19 = v18[1];
        *(_QWORD *)v18[2] = v19;
        if ( v19 )
          *(_QWORD *)(v19 + 16) = v18[2];
      }
      *v18 = a3;
      if ( a3 )
      {
        v20 = *(_QWORD *)(a3 + 16);
        v18[1] = v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = v18 + 1;
        v18[2] = a3 + 16;
        *(_QWORD *)(a3 + 16) = v18;
      }
      ++v16;
    }
    while ( v16 != v17 );
  }
  v21 = *(_QWORD *)(a2 + 16);
  v67 = 0;
  v68 = v72;
  v69 = 16;
  v70 = 0;
  v71 = 1;
  if ( !v21 )
  {
LABEL_28:
    if ( *(_BYTE *)a2 <= 0x1Cu )
      goto LABEL_22;
    if ( !sub_F50EE0((unsigned __int8 *)a2, 0) )
      goto LABEL_30;
    v22 = v66;
    v66[0] = a2;
    v64 = v66;
    v65 = 0x1000000001LL;
    v23 = 1;
    while ( 1 )
    {
      v24 = v22[v23 - 1];
      LODWORD(v65) = v23 - 1;
      if ( (*(_BYTE *)(v24 + 7) & 0x40) != 0 )
      {
        v25 = *(_QWORD *)(v24 - 8);
        v26 = v25 + 32LL * (*(_DWORD *)(v24 + 4) & 0x7FFFFFF);
      }
      else
      {
        v26 = v24;
        v25 = v24 - 32LL * (*(_DWORD *)(v24 + 4) & 0x7FFFFFF);
      }
      for ( i = v25; v26 != i; LODWORD(v65) = v65 + 1 )
      {
        while ( 1 )
        {
          v28 = *(_QWORD *)i;
          if ( *(_QWORD *)i )
          {
            v29 = *(_QWORD *)(i + 8);
            **(_QWORD **)(i + 16) = v29;
            if ( v29 )
              *(_QWORD *)(v29 + 16) = *(_QWORD *)(i + 16);
          }
          *(_QWORD *)i = 0;
          if ( !*(_QWORD *)(v28 + 16) && *(_BYTE *)v28 > 0x1Cu && sub_F50EE0((unsigned __int8 *)v28, 0) )
            break;
          i += 32;
          if ( v26 == i )
            goto LABEL_47;
        }
        v32 = (unsigned int)v65;
        v33 = (unsigned int)v65 + 1LL;
        if ( v33 > HIDWORD(v65) )
        {
          sub_C8D5F0((__int64)&v64, v66, v33, 8u, v30, v31);
          v32 = (unsigned int)v65;
        }
        i += 32;
        v64[v32] = v28;
      }
LABEL_47:
      sub_AE6EC0((__int64)&v67, v24);
      sub_B43D10((_QWORD *)v24);
      v23 = v65;
      if ( !(_DWORD)v65 )
        break;
      v22 = v64;
    }
    if ( v64 != v66 )
      _libc_free((unsigned __int64)v64);
    v34 = (unsigned __int64)(v71 ? &v68[8 * HIDWORD(v69)] : &v68[8 * (unsigned int)v69]);
    v57 = (unsigned __int64)v68;
    v58 = (__int64 *)v34;
    sub_254BBF0((__int64)&v57);
    v59 = &v67;
    v60 = v67;
    v35 = (unsigned __int64)(v71 ? &v68[8 * HIDWORD(v69)] : &v68[8 * (unsigned int)v69]);
    v64 = (__int64 *)v35;
    v65 = v35;
    sub_254BBF0((__int64)&v64);
    v36 = (__int64 *)v57;
    v66[0] = &v67;
    v66[1] = v67;
    if ( v64 == (__int64 *)v57 )
    {
LABEL_30:
      if ( !v71 )
        _libc_free((unsigned __int64)v68);
      goto LABEL_22;
    }
    while ( 1 )
    {
      v37 = *v36;
      v38 = *(_BYTE **)(a1 + 288);
      v53 = *v36;
      if ( v38 == *(_BYTE **)(a1 + 296) )
      {
        sub_24454E0(a1 + 280, v38, &v53);
        v37 = v53;
      }
      else
      {
        if ( v38 )
        {
          *(_QWORD *)v38 = v37;
          v38 = *(_BYTE **)(a1 + 288);
        }
        *(_QWORD *)(a1 + 288) = v38 + 8;
      }
      v54 = v37;
      if ( !v37 )
        goto LABEL_68;
      v39 = *(_DWORD *)(a1 + 96);
      v40 = *(_QWORD *)(a1 + 80);
      if ( !v39 )
        goto LABEL_68;
      v41 = v39 - 1;
      v42 = (v39 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
      v43 = *(_QWORD *)(v40 + 16LL * v42);
      if ( v37 != v43 )
      {
        v52 = 1;
        while ( v43 != -4096 )
        {
          v42 = v41 & (v52 + v42);
          v43 = *(_QWORD *)(v40 + 16LL * v42);
          if ( v37 == v43 )
            goto LABEL_64;
          ++v52;
        }
        goto LABEL_68;
      }
LABEL_64:
      if ( !(unsigned __int8)sub_2D06590(a1 + 72, &v54, &v55) )
        break;
      v44 = v55 + 1;
LABEL_66:
      v45 = *v44;
      if ( v45 )
        *(_BYTE *)(v45 + 41) = 1;
LABEL_68:
      v36 = v58;
      v46 = (__int64 *)(v57 + 8);
      v57 = (unsigned __int64)v46;
      if ( v46 == v58 )
      {
LABEL_71:
        if ( v64 == v58 )
          goto LABEL_30;
      }
      else
      {
        while ( (unsigned __int64)(*v46 + 2) <= 1 )
        {
          v57 = (unsigned __int64)++v46;
          if ( v46 == v58 )
            goto LABEL_71;
        }
        v36 = (__int64 *)v57;
        if ( v64 == (__int64 *)v57 )
          goto LABEL_30;
      }
    }
    v47 = *(_DWORD *)(a1 + 88);
    v48 = *(_DWORD *)(a1 + 96);
    v49 = v55;
    ++*(_QWORD *)(a1 + 72);
    v50 = v47 + 1;
    v56 = v49;
    if ( 4 * (v47 + 1) >= 3 * v48 )
    {
      v48 *= 2;
    }
    else if ( v48 - *(_DWORD *)(a1 + 92) - v50 > v48 >> 3 )
    {
LABEL_77:
      *(_DWORD *)(a1 + 88) = v50;
      if ( *v49 != -4096 )
        --*(_DWORD *)(a1 + 92);
      v51 = v54;
      v49[1] = 0;
      v44 = v49 + 1;
      *(v44 - 1) = v51;
      goto LABEL_66;
    }
    sub_2D08C60(a1 + 72, v48);
    sub_2D06590(a1 + 72, &v54, &v56);
    v50 = *(_DWORD *)(a1 + 88) + 1;
    v49 = v56;
    goto LABEL_77;
  }
LABEL_22:
  if ( v61 != v63 )
    _libc_free((unsigned __int64)v61);
}
