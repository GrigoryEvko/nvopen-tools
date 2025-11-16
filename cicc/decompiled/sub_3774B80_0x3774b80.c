// Function: sub_3774B80
// Address: 0x3774b80
//
unsigned __int8 *__fastcall sub_3774B80(unsigned int *a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, __m128i a6)
{
  __int64 v6; // r15
  __int64 v8; // rdx
  unsigned int v9; // r9d
  _QWORD *v10; // rdi
  _QWORD *v11; // rax
  __int64 v12; // r8
  __int64 v13; // r9
  _QWORD *v14; // r13
  int v15; // edx
  int v16; // ebx
  unsigned __int64 v17; // rdx
  unsigned int v18; // eax
  _QWORD *v19; // rcx
  _BYTE *v20; // r12
  unsigned int v21; // r13d
  __int16 v22; // r15
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int16 v27; // cx
  __int64 v28; // rax
  __int64 v29; // rbx
  _BYTE *v30; // r12
  __int64 v31; // rax
  __int64 v32; // rdx
  unsigned __int64 v33; // r8
  char v34; // r11
  __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  __int64 v40; // rax
  unsigned __int8 *v41; // rax
  int v42; // edx
  int v43; // edi
  unsigned __int8 *v44; // rdx
  __int64 v45; // rax
  unsigned __int8 *v46; // r12
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rcx
  _BYTE *v51; // rdx
  __int128 v52; // [rsp-10h] [rbp-130h]
  __int64 v53; // [rsp+8h] [rbp-118h]
  __int64 v55; // [rsp+30h] [rbp-F0h]
  unsigned __int64 v56; // [rsp+38h] [rbp-E8h]
  int v57; // [rsp+38h] [rbp-E8h]
  unsigned __int16 v58; // [rsp+44h] [rbp-DCh]
  char v59; // [rsp+44h] [rbp-DCh]
  int v60; // [rsp+44h] [rbp-DCh]
  unsigned int v61; // [rsp+48h] [rbp-D8h]
  unsigned __int64 v62; // [rsp+48h] [rbp-D8h]
  __int64 v64; // [rsp+58h] [rbp-C8h]
  unsigned __int16 v65; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v66; // [rsp+78h] [rbp-A8h]
  __int16 v67; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v68; // [rsp+88h] [rbp-98h]
  __int64 v69; // [rsp+90h] [rbp-90h]
  __int64 v70; // [rsp+98h] [rbp-88h]
  __int64 v71; // [rsp+A0h] [rbp-80h]
  __int64 v72; // [rsp+A8h] [rbp-78h]
  _BYTE *v73; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v74; // [rsp+B8h] [rbp-68h]
  _BYTE v75[96]; // [rsp+C0h] [rbp-60h] BYREF

  v8 = *((unsigned __int16 *)a1 + 8);
  if ( (_WORD)v8 )
  {
    v64 = 0;
    v9 = (unsigned __int16)word_4456580[(int)v8 - 1];
  }
  else
  {
    v48 = sub_3009970((__int64)(a1 + 4), (__int64)a2, v8, a4, a5);
    v64 = v49;
    v6 = v48;
    v9 = v48;
  }
  LOWORD(v6) = v9;
  v10 = (_QWORD *)*((_QWORD *)a1 + 1);
  v61 = v9;
  v73 = 0;
  LODWORD(v74) = 0;
  v11 = sub_33F17F0(v10, 51, (__int64)&v73, v6, v64);
  v13 = v61;
  v14 = v11;
  v16 = v15;
  if ( v73 )
  {
    sub_B91220((__int64)&v73, (__int64)v73);
    v13 = v61;
  }
  v17 = *a1;
  v73 = v75;
  v18 = v17;
  v74 = 0x300000000LL;
  if ( (unsigned int)v17 > 3 )
  {
    v57 = v17;
    v60 = v13;
    v62 = v17;
    sub_C8D5F0((__int64)&v73, v75, v17, 0x10u, v12, v13);
    v50 = (__int64)v73;
    LODWORD(v13) = v60;
    v51 = &v73[16 * v62];
    do
    {
      if ( v50 )
      {
        *(_QWORD *)v50 = v14;
        *(_DWORD *)(v50 + 8) = v16;
      }
      v50 += 16;
    }
    while ( (_BYTE *)v50 != v51 );
    LODWORD(v74) = v57;
    v20 = v73;
    v18 = *a1;
  }
  else
  {
    v19 = v75;
    v20 = v75;
    if ( v17 )
    {
      do
      {
        *v19 = v14;
        v19 += 2;
        *((_DWORD *)v19 - 2) = v16;
        --v17;
      }
      while ( v17 );
      v20 = v73;
    }
    LODWORD(v74) = v18;
  }
  if ( v18 )
  {
    v55 = v6;
    v21 = 0;
    v22 = v13;
    v53 = 16LL * ((unsigned __int16)v13 - 1) + 71615648;
    while ( 1 )
    {
      v23 = *(unsigned int *)(a4 + 4LL * v21);
      if ( (_DWORD)v23 == -1 )
        goto LABEL_14;
      v29 = 16LL * v21;
      v30 = &v20[v29];
      if ( (unsigned int)v23 >= v18 )
      {
        v23 = (unsigned int)v23 - v18;
        v24 = *a3;
      }
      else
      {
        v24 = *a2;
      }
      v25 = *(_QWORD *)(v24 + 40) + 40 * v23;
      *(_QWORD *)v30 = *(_QWORD *)v25;
      *((_DWORD *)v30 + 2) = *(_DWORD *)(v25 + 8);
      v20 = v73;
      v26 = *(_QWORD *)(*(_QWORD *)&v73[16 * v21] + 48LL) + 16LL * *(unsigned int *)&v73[16 * v21 + 8];
      v27 = *(_WORD *)v26;
      v28 = *(_QWORD *)(v26 + 8);
      v65 = v27;
      v66 = v28;
      if ( v27 != v22 )
        break;
      if ( !v22 && v28 != v64 )
      {
        v67 = 0;
        v68 = v64;
LABEL_21:
        v58 = v27;
        v31 = sub_3007260((__int64)&v67);
        v27 = v58;
        LODWORD(v13) = v32;
        v71 = v31;
        v33 = v31;
        v72 = v32;
        v34 = v32;
        goto LABEL_22;
      }
LABEL_13:
      v18 = *a1;
LABEL_14:
      if ( v18 <= ++v21 )
        goto LABEL_36;
    }
    v67 = v22;
    v68 = v64;
    if ( !v22 )
      goto LABEL_21;
    if ( v22 == 1 || (unsigned __int16)(v22 - 504) <= 7u )
LABEL_45:
      BUG();
    v33 = *(_QWORD *)v53;
    v34 = *(_BYTE *)(v53 + 8);
LABEL_22:
    if ( v27 )
    {
      if ( v27 == 1 || (unsigned __int16)(v27 - 504) <= 7u )
        goto LABEL_45;
      v39 = *(_QWORD *)&byte_444C4A0[16 * v27 - 16];
      LOBYTE(v38) = byte_444C4A0[16 * v27 - 8];
    }
    else
    {
      v56 = v33;
      v59 = v34;
      v35 = sub_3007260((__int64)&v65);
      v33 = v56;
      v34 = v59;
      v36 = v35;
      v38 = v37;
      v69 = v36;
      v39 = v36;
      v70 = v38;
    }
    if ( ((_BYTE)v38 || !v34) && v39 > v33 )
    {
      v40 = v55;
      LOWORD(v40) = v22;
      v55 = v40;
      v41 = sub_33FAF80(*((_QWORD *)a1 + 1), 216, *((_QWORD *)a1 + 4), (unsigned int)v40, v64, v13, a6);
      v43 = v42;
      v44 = v41;
      v45 = (__int64)v73;
      *(_QWORD *)&v73[16 * v21] = v44;
      *(_DWORD *)(v45 + v29 + 8) = v43;
      v20 = v73;
    }
    goto LABEL_13;
  }
LABEL_36:
  *((_QWORD *)&v52 + 1) = (unsigned int)v74;
  *(_QWORD *)&v52 = v20;
  v46 = sub_33FC220(
          *((_QWORD **)a1 + 1),
          156,
          *((_QWORD *)a1 + 4),
          *((_QWORD *)a1 + 2),
          *((_QWORD *)a1 + 3),
          *((_QWORD *)a1 + 4),
          v52);
  if ( v73 != v75 )
    _libc_free((unsigned __int64)v73);
  return v46;
}
