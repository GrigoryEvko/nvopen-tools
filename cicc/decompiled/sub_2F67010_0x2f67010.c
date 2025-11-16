// Function: sub_2F67010
// Address: 0x2f67010
//
void __fastcall sub_2F67010(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _DWORD *a5, _DWORD *a6)
{
  _DWORD *v6; // r15
  int v11; // esi
  int v12; // ecx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  int v17; // r13d
  _OWORD *v18; // rax
  _BYTE *v19; // rcx
  _OWORD *i; // rdx
  int v21; // ecx
  __int64 v22; // rax
  int v23; // esi
  __int64 v24; // rdx
  unsigned __int64 v25; // r13
  __int64 v26; // rax
  int v27; // ebx
  __int64 v28; // rbx
  __int64 v29; // r13
  __int64 v30; // r13
  __int64 v31; // rbx
  __int64 v32; // r9
  unsigned int v33; // eax
  __int64 v34; // rcx
  unsigned int v35; // eax
  unsigned int v36; // eax
  unsigned int v37; // ecx
  __int64 v38; // rdx
  unsigned int v39; // eax
  __int64 v40; // rsi
  unsigned int v41; // eax
  unsigned int v42; // eax
  unsigned int v43; // esi
  __int64 v44; // rdi
  __int64 v45; // r8
  __int64 v46; // r9
  _BYTE *v47; // rcx
  _OWORD *v48; // rax
  _OWORD *j; // rdx
  unsigned __int64 v51; // [rsp+38h] [rbp-648h]
  unsigned __int64 v52; // [rsp+38h] [rbp-648h]
  __int64 *v54; // [rsp+50h] [rbp-630h] BYREF
  __int64 v55; // [rsp+58h] [rbp-628h]
  _BYTE v56[64]; // [rsp+60h] [rbp-620h] BYREF
  unsigned __int64 v57[2]; // [rsp+A0h] [rbp-5E0h] BYREF
  _BYTE v58[128]; // [rsp+B0h] [rbp-5D0h] BYREF
  __int64 v59; // [rsp+130h] [rbp-550h] BYREF
  int v60; // [rsp+138h] [rbp-548h]
  int v61; // [rsp+13Ch] [rbp-544h]
  __int64 v62; // [rsp+140h] [rbp-540h]
  _DWORD *v63; // [rsp+148h] [rbp-538h]
  __int16 v64; // [rsp+150h] [rbp-530h]
  unsigned __int64 *v65; // [rsp+158h] [rbp-528h]
  _DWORD *v66; // [rsp+160h] [rbp-520h]
  __int64 v67; // [rsp+168h] [rbp-518h]
  __int64 v68; // [rsp+170h] [rbp-510h]
  __int64 v69; // [rsp+178h] [rbp-508h]
  void *v70; // [rsp+180h] [rbp-500h] BYREF
  __int64 v71; // [rsp+188h] [rbp-4F8h]
  _DWORD v72[8]; // [rsp+190h] [rbp-4F0h] BYREF
  _BYTE *v73; // [rsp+1B0h] [rbp-4D0h] BYREF
  __int64 v74; // [rsp+1B8h] [rbp-4C8h]
  _BYTE v75[512]; // [rsp+1C0h] [rbp-4C0h] BYREF
  __int64 v76; // [rsp+3C0h] [rbp-2C0h] BYREF
  int v77; // [rsp+3C8h] [rbp-2B8h]
  int v78; // [rsp+3CCh] [rbp-2B4h]
  __int64 v79; // [rsp+3D0h] [rbp-2B0h]
  _DWORD *v80; // [rsp+3D8h] [rbp-2A8h]
  __int16 v81; // [rsp+3E0h] [rbp-2A0h]
  unsigned __int64 *v82; // [rsp+3E8h] [rbp-298h]
  _DWORD *v83; // [rsp+3F0h] [rbp-290h]
  __int64 v84; // [rsp+3F8h] [rbp-288h]
  __int64 v85; // [rsp+400h] [rbp-280h]
  __int64 v86; // [rsp+408h] [rbp-278h]
  void *s; // [rsp+410h] [rbp-270h] BYREF
  __int64 v88; // [rsp+418h] [rbp-268h]
  _DWORD v89[8]; // [rsp+420h] [rbp-260h] BYREF
  _BYTE *v90; // [rsp+440h] [rbp-240h] BYREF
  __int64 v91; // [rsp+448h] [rbp-238h]
  _BYTE v92[560]; // [rsp+450h] [rbp-230h] BYREF

  v6 = a5;
  v11 = a6[3];
  v12 = a6[5];
  v57[0] = (unsigned __int64)v58;
  v57[1] = 0x1000000000LL;
  v13 = *(_QWORD *)(a1 + 40);
  v60 = v11;
  v14 = *(_QWORD *)(a1 + 24);
  v64 = 257;
  v65 = v57;
  v66 = a6;
  v67 = v13;
  v15 = *(_QWORD *)(v13 + 32);
  v69 = v14;
  v16 = *(unsigned int *)(a3 + 72);
  v68 = v15;
  v17 = v16;
  v70 = v72;
  v59 = a3;
  v61 = v12;
  v62 = a4;
  v63 = a5;
  v71 = 0x800000000LL;
  if ( (unsigned int)v16 > 8 )
  {
    v51 = v16;
    sub_C8D5F0((__int64)&v70, v72, v16, 4u, (__int64)a5, (__int64)a6);
    memset(v70, 255, 4 * v51);
    LODWORD(v71) = v17;
    v16 = *(unsigned int *)(a3 + 72);
    v17 = *(_DWORD *)(a3 + 72);
  }
  else
  {
    if ( v16 )
    {
      v39 = 4 * v16;
      if ( 4 * v16 )
      {
        if ( v39 < 8 )
        {
          if ( (v39 & 4) != 0 )
          {
            v72[0] = -1;
            v72[v39 / 4 - 1] = -1;
          }
          else if ( v39 )
          {
            LOBYTE(v72[0]) = -1;
          }
        }
        else
        {
          a5 = v72;
          v40 = v39;
          v41 = v39 - 1;
          *(_QWORD *)((char *)&v72[-2] + v40) = -1;
          if ( v41 >= 8 )
          {
            v42 = v41 & 0xFFFFFFF8;
            v43 = 0;
            do
            {
              v44 = v43;
              v43 += 8;
              *(_QWORD *)((char *)v72 + v44) = -1;
            }
            while ( v43 < v42 );
          }
        }
      }
    }
    LODWORD(v71) = v16;
  }
  v18 = v75;
  v73 = v75;
  v74 = 0x800000000LL;
  if ( v16 )
  {
    v19 = v75;
    if ( v16 > 8 )
    {
      v52 = v16;
      sub_C8D5F0((__int64)&v73, v75, v16, 0x40u, (__int64)a5, (__int64)a6);
      v19 = v73;
      v16 = v52;
      v18 = &v73[64 * (unsigned __int64)(unsigned int)v74];
    }
    for ( i = &v19[64 * v16]; i != v18; v18 += 4 )
    {
      if ( v18 )
      {
        *v18 = 0;
        v18[1] = 0;
        v18[2] = 0;
        v18[3] = 0;
      }
    }
    LODWORD(v74) = v17;
  }
  v21 = a6[4];
  v22 = *(_QWORD *)(a1 + 40);
  v83 = a6;
  v23 = a6[2];
  v79 = a4;
  v78 = v21;
  v24 = *(_QWORD *)(a1 + 24);
  v81 = 257;
  v84 = v22;
  v25 = *(unsigned int *)(a2 + 72);
  v82 = v57;
  v26 = *(_QWORD *)(v22 + 32);
  v76 = a2;
  v27 = v25;
  v85 = v26;
  v77 = v23;
  v80 = v6;
  v86 = v24;
  s = v89;
  v88 = 0x800000000LL;
  if ( (unsigned int)v25 > 8 )
  {
    sub_C8D5F0((__int64)&s, v89, v25, 4u, (__int64)a5, (__int64)a6);
    memset(s, 255, 4 * v25);
    LODWORD(v88) = v25;
    v91 = 0x800000000LL;
    v25 = *(unsigned int *)(a2 + 72);
    v90 = v92;
    v27 = v25;
    if ( !v25 )
      goto LABEL_15;
    if ( v25 > 8 )
    {
      sub_C8D5F0((__int64)&v90, v92, v25, 0x40u, v45, v46);
      v47 = v90;
      v48 = &v90[64 * (unsigned __int64)(unsigned int)v91];
      goto LABEL_57;
    }
LABEL_56:
    v47 = v92;
    v48 = v92;
LABEL_57:
    for ( j = &v47[64 * v25]; j != v48; v48 += 4 )
    {
      if ( v48 )
      {
        *v48 = 0;
        v48[1] = 0;
        v48[2] = 0;
        v48[3] = 0;
      }
    }
    LODWORD(v91) = v27;
    goto LABEL_15;
  }
  if ( v25 )
  {
    v33 = 4 * v25;
    if ( 4 * v25 )
    {
      if ( v33 < 8 )
      {
        if ( (v33 & 4) != 0 )
        {
          v89[0] = -1;
          v89[v33 / 4 - 1] = -1;
        }
        else if ( v33 )
        {
          LOBYTE(v89[0]) = -1;
        }
      }
      else
      {
        v34 = v33;
        v35 = v33 - 1;
        *(_QWORD *)((char *)&v89[-2] + v34) = -1;
        if ( v35 >= 8 )
        {
          v36 = v35 & 0xFFFFFFF8;
          v37 = 0;
          do
          {
            v38 = v37;
            v37 += 8;
            *(_QWORD *)((char *)v89 + v38) = -1;
          }
          while ( v37 < v36 );
        }
      }
    }
  }
  LODWORD(v88) = v25;
  v90 = v92;
  v91 = 0x800000000LL;
  if ( v25 )
    goto LABEL_56;
LABEL_15:
  v28 = *(unsigned int *)(v76 + 72);
  if ( (_DWORD)v28 )
  {
    v29 = 0;
    while ( 1 )
    {
      sub_2F66F20(&v76, v29, (__int64)&v59);
      if ( *(_DWORD *)&v90[64 * v29] == 5 )
        break;
      if ( ++v29 == v28 )
        goto LABEL_20;
    }
LABEL_72:
    BUG();
  }
LABEL_20:
  v30 = 0;
  v31 = *(unsigned int *)(v59 + 72);
  if ( (_DWORD)v31 )
  {
    do
    {
      sub_2F66F20(&v59, v30, (__int64)&v76);
      if ( *(_DWORD *)&v73[64 * v30] == 5 )
        goto LABEL_72;
    }
    while ( ++v30 != v31 );
  }
  if ( !(unsigned __int8)sub_2F62F70(&v76, (__int64)&v59) || !(unsigned __int8)sub_2F62F70(&v59, (__int64)&v76) )
    goto LABEL_72;
  v55 = 0x800000000LL;
  v54 = (__int64 *)v56;
  sub_2F623C0(&v76, &v59, (__int64)&v54, 0);
  sub_2F623C0(&v59, &v76, (__int64)&v54, 0);
  sub_2F60CF0(&v76);
  sub_2F60CF0(&v59);
  sub_2E0F950(a2, (__int64 *)a3, (__int64)s, (__int64)v70, (__int64)v57, v32);
  if ( (_DWORD)v55 )
    sub_2E12C90(*(_QWORD **)(a1 + 40), a2, v54, (unsigned int)v55, 0, 0);
  if ( v54 != (__int64 *)v56 )
    _libc_free((unsigned __int64)v54);
  if ( v90 != v92 )
    _libc_free((unsigned __int64)v90);
  if ( s != v89 )
    _libc_free((unsigned __int64)s);
  if ( v73 != v75 )
    _libc_free((unsigned __int64)v73);
  if ( v70 != v72 )
    _libc_free((unsigned __int64)v70);
  if ( (_BYTE *)v57[0] != v58 )
    _libc_free(v57[0]);
}
