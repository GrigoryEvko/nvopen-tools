// Function: sub_2AC4ED0
// Address: 0x2ac4ed0
//
__int64 __fastcall sub_2AC4ED0(__int64 a1, unsigned __int8 *a2, const void *a3, unsigned __int64 a4, __int64 a5)
{
  char v7; // bl
  int v8; // eax
  __int64 v9; // r9
  int v10; // ebx
  int v11; // edx
  __int64 v12; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  unsigned __int64 v19; // r15
  signed __int64 v20; // r15
  __int64 v21; // r8
  _QWORD *v22; // rax
  __int64 v23; // rsi
  char v24; // bl
  __int64 v25; // rdx
  __int64 v26; // rcx
  unsigned __int64 v27; // r8
  __int64 v28; // r9
  _QWORD *v29; // r14
  __int64 v30; // r12
  __int64 v31; // rax
  __int64 v32; // r9
  __int64 v33; // rbx
  _QWORD *v34; // rdi
  unsigned __int64 v35; // rdx
  __int64 v36; // r12
  __int64 v37; // rax
  __int64 v38; // r9
  unsigned __int8 *v39; // r14
  __int64 v40; // r8
  __int64 v41; // r12
  unsigned __int64 v42; // rax
  __int64 v43; // r13
  __int64 v44; // rdi
  int v45; // edx
  char *v46; // rbx
  _QWORD *v47; // rsi
  char *v48; // rcx
  __int64 v49; // r12
  _QWORD *v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // [rsp+8h] [rbp-B8h]
  int v54; // [rsp+8h] [rbp-B8h]
  int v55; // [rsp+8h] [rbp-B8h]
  void *srcb; // [rsp+10h] [rbp-B0h]
  _QWORD *srca; // [rsp+10h] [rbp-B0h]
  __int64 v60; // [rsp+18h] [rbp-A8h]
  __int64 v61; // [rsp+20h] [rbp-A0h]
  unsigned __int8 *v62; // [rsp+28h] [rbp-98h] BYREF
  __int64 v63; // [rsp+30h] [rbp-90h] BYREF
  __int64 v64; // [rsp+38h] [rbp-88h] BYREF
  __int64 v65[2]; // [rsp+40h] [rbp-80h] BYREF
  void *v66; // [rsp+50h] [rbp-70h]
  void *v67; // [rsp+58h] [rbp-68h]
  _QWORD *v68; // [rsp+60h] [rbp-60h] BYREF
  __int64 v69; // [rsp+68h] [rbp-58h]
  _QWORD v70[10]; // [rsp+70h] [rbp-50h] BYREF

  v62 = a2;
  v68 = (_QWORD *)a1;
  v69 = (__int64)a2;
  v70[1] = sub_2AC3AD0;
  v70[0] = sub_2AA7CF0;
  v7 = sub_2BF1270(&v68, a5);
  sub_A17130((__int64)&v68);
  if ( v7 )
    return 0;
  v8 = sub_9B78C0((__int64)v62, *(__int64 **)(a1 + 16));
  v10 = v8;
  if ( v8 )
  {
    if ( (unsigned int)(v8 - 210) <= 1 || v8 == 11 || v8 == 324 || v8 == 291 || v8 == 155 )
      return 0;
  }
  v11 = *v62;
  if ( v11 == 40 )
  {
    v61 = 32LL * (unsigned int)sub_B491D0((__int64)v62);
  }
  else
  {
    v61 = 0;
    if ( v11 != 85 )
    {
      v61 = 64;
      if ( v11 != 34 )
        BUG();
    }
  }
  if ( (v62[7] & 0x80u) == 0 )
    goto LABEL_22;
  v14 = sub_BD2BC0((__int64)v62);
  v53 = v15 + v14;
  if ( (v62[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v53 >> 4) )
LABEL_60:
      BUG();
LABEL_22:
    v18 = 0;
    goto LABEL_23;
  }
  if ( !(unsigned int)((v53 - sub_BD2BC0((__int64)v62)) >> 4) )
    goto LABEL_22;
  if ( (v62[7] & 0x80u) == 0 )
    goto LABEL_60;
  v54 = *(_DWORD *)(sub_BD2BC0((__int64)v62) + 8);
  if ( (v62[7] & 0x80u) == 0 )
    BUG();
  v16 = sub_BD2BC0((__int64)v62);
  v18 = 32LL * (unsigned int)(*(_DWORD *)(v16 + v17 - 4) - v54);
LABEL_23:
  v19 = (unsigned int)((32LL * (*((_DWORD *)v62 + 1) & 0x7FFFFFF) - 32 - v61 - v18) >> 5);
  if ( v19 > a4 )
    v19 = a4;
  v20 = 8 * v19;
  v68 = v70;
  v69 = 0x400000000LL;
  v21 = v20 >> 3;
  if ( (unsigned __int64)v20 > 0x20 )
  {
    sub_C8D5F0((__int64)&v68, v70, v20 >> 3, 8u, v21, v9);
    v21 = v20 >> 3;
    v34 = &v68[(unsigned int)v69];
  }
  else
  {
    if ( !v20 )
    {
      LODWORD(v69) = 0;
      if ( !v10 )
        goto LABEL_28;
      goto LABEL_41;
    }
    v34 = v70;
  }
  v55 = v21;
  memcpy(v34, a3, v20);
  LODWORD(v69) = v69 + v55;
  if ( !v10 )
    goto LABEL_28;
LABEL_41:
  v65[0] = a1;
  v65[1] = (__int64)&v62;
  v67 = sub_2AAA020;
  v66 = sub_2AA7D20;
  if ( (unsigned __int8)sub_2BF1270(v65, a5) )
  {
    sub_A17130((__int64)v65);
    v35 = (unsigned __int64)v68;
    v36 = (unsigned int)v69;
    v60 = *((_QWORD *)v62 + 1);
    v65[0] = *((_QWORD *)v62 + 6);
    if ( v65[0] )
    {
      srcb = v68;
      sub_2AAAFA0(v65);
      v35 = (unsigned __int64)srcb;
    }
    srca = (_QWORD *)v35;
    v37 = sub_22077B0(0xB8u);
    v12 = v37;
    if ( v37 )
    {
      v39 = v62;
      sub_2ABAD10(v37, 18, srca, v36, v62, v38);
      *(_DWORD *)(v12 + 160) = v10;
      *(_QWORD *)v12 = &unk_4A23D28;
      *(_QWORD *)(v12 + 96) = &unk_4A23DA0;
      *(_QWORD *)(v12 + 40) = &unk_4A23D68;
      *(_QWORD *)(v12 + 168) = v60;
      *(_BYTE *)(v12 + 176) = sub_B46420((__int64)v39);
      *(_BYTE *)(v12 + 177) = sub_B46490((__int64)v39);
      *(_BYTE *)(v12 + 178) = sub_B46970(v39);
    }
    goto LABEL_36;
  }
  sub_A17130((__int64)v65);
LABEL_28:
  v66 = 0;
  v63 = 0;
  v64 = 0;
  v22 = (_QWORD *)sub_22077B0(0x20u);
  if ( v22 )
  {
    v22[1] = a1;
    *v22 = &v63;
    v22[2] = &v62;
    v22[3] = &v64;
  }
  v65[0] = (__int64)v22;
  v23 = a5;
  v67 = sub_2AAA550;
  v12 = 0;
  v66 = sub_2AA8880;
  v24 = sub_2BF1270(v65, v23);
  sub_A17130((__int64)v65);
  if ( !v24 )
    goto LABEL_37;
  if ( BYTE4(v64) )
  {
    if ( (unsigned __int8)sub_B19060(*(_QWORD *)(a1 + 32) + 440LL, (__int64)v62, v25, v26) )
    {
      v41 = sub_2AB6F10(a1, *((_QWORD *)v62 + 5));
    }
    else
    {
      v49 = *(_QWORD *)a1;
      v50 = (_QWORD *)sub_BD5C60((__int64)v62);
      v51 = sub_BCB2A0(v50);
      v52 = sub_AD6400(v51);
      v41 = sub_2AC42A0(v49, v52);
    }
    v42 = (unsigned __int64)v68;
    v43 = (unsigned int)v64;
    v44 = (unsigned int)v69;
    v45 = v69;
    v46 = (char *)&v68[v43];
    v47 = &v68[v44];
    if ( &v68[v43] == &v68[v44] )
    {
      sub_2AB9420((__int64)&v68, v41, (unsigned int)v69, (unsigned int)v69, v40, v28);
    }
    else
    {
      v27 = (unsigned int)v69 + 1LL;
      if ( v27 > HIDWORD(v69) )
      {
        sub_C8D5F0((__int64)&v68, v70, v27, 8u, v27, v28);
        v42 = (unsigned __int64)v68;
        v45 = v69;
        v44 = (unsigned int)v69;
        v46 = (char *)&v68[v43];
        v47 = &v68[v44];
      }
      v48 = (char *)(v42 + v44 * 8 - 8);
      if ( v47 )
      {
        *v47 = *(_QWORD *)v48;
        v42 = (unsigned __int64)v68;
        v45 = v69;
        v44 = (unsigned int)v69;
        v48 = (char *)&v68[v44 - 1];
      }
      if ( v46 != v48 )
      {
        memmove((void *)(v42 + v44 * 8 - (v48 - v46)), v46, v48 - v46);
        v45 = v69;
      }
      v25 = (unsigned int)(v45 + 1);
      LODWORD(v69) = v25;
      *(_QWORD *)v46 = v41;
    }
  }
  sub_2AB9420((__int64)&v68, *((_QWORD *)a3 + a4 - 1), v25, a4, v27, v28);
  v29 = v68;
  v30 = (unsigned int)v69;
  v65[0] = *((_QWORD *)v62 + 6);
  if ( v65[0] )
    sub_2AAAFA0(v65);
  v31 = sub_22077B0(0xA8u);
  v12 = v31;
  if ( v31 )
  {
    v33 = v63;
    sub_2ABAD10(v31, 14, v29, v30, v62, v32);
    *(_QWORD *)(v12 + 160) = v33;
    *(_QWORD *)v12 = &unk_4A23C98;
    *(_QWORD *)(v12 + 40) = &unk_4A23CD0;
    *(_QWORD *)(v12 + 96) = &unk_4A23D08;
  }
LABEL_36:
  sub_9C6650(v65);
LABEL_37:
  if ( v68 != v70 )
    _libc_free((unsigned __int64)v68);
  return v12;
}
