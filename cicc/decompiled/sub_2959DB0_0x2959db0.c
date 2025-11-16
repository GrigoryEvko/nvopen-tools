// Function: sub_2959DB0
// Address: 0x2959db0
//
void __fastcall sub_2959DB0(
        __int64 a1,
        unsigned __int8 **a2,
        __int64 a3,
        char a4,
        __int64 a5,
        __int64 a6,
        char a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  unsigned __int8 **v10; // r15
  unsigned __int8 **v11; // r13
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned int v16; // ebx
  unsigned __int8 *v17; // r12
  __int64 v18; // rdx
  _QWORD *v19; // rax
  __int64 v20; // rbx
  __int64 v21; // r12
  _BYTE *v22; // r12
  unsigned __int64 v23; // r13
  __int64 v24; // rdx
  unsigned int v25; // esi
  unsigned __int64 v26; // r13
  __int64 v27; // r14
  __int64 v28; // rax
  unsigned int v29; // r15d
  unsigned __int8 *v30; // r12
  __int64 v31; // rax
  unsigned __int8 *v32; // r14
  __int64 (__fastcall *v33)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  unsigned __int64 v34; // r14
  _BYTE *v35; // rbx
  __int64 v36; // rdx
  unsigned int v37; // esi
  __int64 v38; // rcx
  _QWORD *v39; // rax
  __int64 v40; // r9
  __int64 v41; // rbx
  unsigned __int64 v42; // r13
  _BYTE *v43; // r12
  __int64 v44; // rdx
  unsigned int v45; // esi
  __int64 v46; // rax
  unsigned __int8 *v47; // r12
  unsigned int v48; // r15d
  __int64 v49; // rax
  unsigned __int8 *v50; // r14
  __int64 (__fastcall *v51)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  unsigned __int64 v52; // r14
  _BYTE *v53; // rbx
  __int64 v54; // rdx
  unsigned int v55; // esi
  unsigned __int8 **v59; // [rsp+48h] [rbp-168h]
  unsigned __int64 v60; // [rsp+48h] [rbp-168h]
  _QWORD v61[4]; // [rsp+50h] [rbp-160h] BYREF
  __int16 v62; // [rsp+70h] [rbp-140h]
  _BYTE v63[32]; // [rsp+80h] [rbp-130h] BYREF
  __int16 v64; // [rsp+A0h] [rbp-110h]
  __int64 *v65; // [rsp+B0h] [rbp-100h] BYREF
  __int64 v66; // [rsp+B8h] [rbp-F8h]
  _QWORD v67[6]; // [rsp+C0h] [rbp-F0h] BYREF
  _BYTE *v68; // [rsp+F0h] [rbp-C0h]
  __int64 v69; // [rsp+F8h] [rbp-B8h]
  _BYTE v70[32]; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v71; // [rsp+120h] [rbp-90h]
  __int64 v72; // [rsp+128h] [rbp-88h]
  __int64 v73; // [rsp+130h] [rbp-80h]
  __int64 v74; // [rsp+138h] [rbp-78h]
  void **v75; // [rsp+140h] [rbp-70h]
  void **v76; // [rsp+148h] [rbp-68h]
  __int64 v77; // [rsp+150h] [rbp-60h]
  int v78; // [rsp+158h] [rbp-58h]
  __int16 v79; // [rsp+15Ch] [rbp-54h]
  char v80; // [rsp+15Eh] [rbp-52h]
  __int64 v81; // [rsp+160h] [rbp-50h]
  __int64 v82; // [rsp+168h] [rbp-48h]
  void *v83; // [rsp+170h] [rbp-40h] BYREF
  void *v84; // [rsp+178h] [rbp-38h] BYREF

  v10 = a2;
  v11 = &a2[a3];
  v71 = a1;
  v74 = sub_AA48A0(a1);
  v75 = &v83;
  v76 = &v84;
  v79 = 512;
  v68 = v70;
  v83 = &unk_49DA100;
  v69 = 0x200000000LL;
  LOWORD(v73) = 0;
  v84 = &unk_49DA0B0;
  v65 = v67;
  v66 = 0x600000000LL;
  v77 = 0;
  v78 = 0;
  v80 = 7;
  v81 = 0;
  v82 = 0;
  v72 = a1 + 48;
  if ( a2 != v11 )
  {
    do
    {
      v17 = *v10;
      if ( a7 && !sub_98ED60(*v10, a9, a8, a10, 0) )
      {
        v61[0] = sub_BD5D20((__int64)v17);
        v61[2] = ".fr";
        v62 = 773;
        v61[1] = v18;
        v64 = 257;
        v19 = sub_BD2C40(72, 1u);
        v20 = (__int64)v19;
        if ( v19 )
          sub_B549F0((__int64)v19, (__int64)v17, (__int64)v63, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v76 + 2))(v76, v20, v61, v72, v73);
        v21 = 16LL * (unsigned int)v69;
        if ( v68 != &v68[v21] )
        {
          v59 = v11;
          v22 = &v68[v21];
          v23 = (unsigned __int64)v68;
          do
          {
            v24 = *(_QWORD *)(v23 + 8);
            v25 = *(_DWORD *)v23;
            v23 += 16LL;
            sub_B99FD0(v20, v25, v24);
          }
          while ( v22 != (_BYTE *)v23 );
          v11 = v59;
        }
        v17 = (unsigned __int8 *)v20;
      }
      v14 = (unsigned int)v66;
      v15 = (unsigned int)v66 + 1LL;
      if ( v15 > HIDWORD(v66) )
      {
        sub_C8D5F0((__int64)&v65, v67, v15, 8u, v12, v13);
        v14 = (unsigned int)v66;
      }
      ++v10;
      v65[v14] = (__int64)v17;
      v16 = v66 + 1;
      LODWORD(v66) = v66 + 1;
    }
    while ( v11 != v10 );
    v26 = (unsigned __int64)v65;
    v60 = v16;
    v27 = *v65;
    if ( a4 )
    {
      if ( v16 <= 1uLL )
        goto LABEL_30;
      v28 = 1;
      v29 = 1;
      v30 = (unsigned __int8 *)*v65;
      while ( 1 )
      {
        v62 = 257;
        v32 = *(unsigned __int8 **)(v26 + 8 * v28);
        v33 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v75 + 2);
        if ( v33 != sub_9202E0 )
          break;
        if ( *v30 > 0x15u || *v32 > 0x15u )
        {
LABEL_26:
          v64 = 257;
          v30 = (unsigned __int8 *)sub_B504D0(29, (__int64)v30, (__int64)v32, (__int64)v63, 0, 0);
          (*((void (__fastcall **)(void **, unsigned __int8 *, _QWORD *, __int64, __int64))*v76 + 2))(
            v76,
            v30,
            v61,
            v72,
            v73);
          v34 = (unsigned __int64)v68;
          v35 = &v68[16 * (unsigned int)v69];
          if ( v68 == v35 )
            goto LABEL_23;
          do
          {
            v36 = *(_QWORD *)(v34 + 8);
            v37 = *(_DWORD *)v34;
            v34 += 16LL;
            sub_B99FD0((__int64)v30, v37, v36);
          }
          while ( v35 != (_BYTE *)v34 );
          v28 = ++v29;
          if ( v29 >= v60 )
          {
LABEL_29:
            v27 = (__int64)v30;
            goto LABEL_30;
          }
        }
        else
        {
          if ( (unsigned __int8)sub_AC47B0(29) )
            v31 = sub_AD5570(29, (__int64)v30, v32, 0, 0);
          else
            v31 = sub_AABE40(0x1Du, v30, v32);
LABEL_21:
          if ( !v31 )
            goto LABEL_26;
          v30 = (unsigned __int8 *)v31;
LABEL_23:
          v28 = ++v29;
          if ( v29 >= v60 )
            goto LABEL_29;
        }
      }
      v31 = v33((__int64)v75, 29u, v30, v32);
      goto LABEL_21;
    }
    if ( v16 <= 1uLL )
      goto LABEL_31;
    v46 = 1;
    v47 = (unsigned __int8 *)*v65;
    v48 = 1;
    while ( 1 )
    {
      v62 = 257;
      v50 = *(unsigned __int8 **)(v26 + 8 * v46);
      v51 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v75 + 2);
      if ( v51 != sub_9202E0 )
        break;
      if ( *v47 > 0x15u || *v50 > 0x15u )
      {
LABEL_50:
        v64 = 257;
        v47 = (unsigned __int8 *)sub_B504D0(28, (__int64)v47, (__int64)v50, (__int64)v63, 0, 0);
        (*((void (__fastcall **)(void **, unsigned __int8 *, _QWORD *, __int64, __int64))*v76 + 2))(
          v76,
          v47,
          v61,
          v72,
          v73);
        v52 = (unsigned __int64)v68;
        v53 = &v68[16 * (unsigned int)v69];
        if ( v68 == v53 )
          goto LABEL_47;
        do
        {
          v54 = *(_QWORD *)(v52 + 8);
          v55 = *(_DWORD *)v52;
          v52 += 16LL;
          sub_B99FD0((__int64)v47, v55, v54);
        }
        while ( v53 != (_BYTE *)v52 );
        v46 = ++v48;
        if ( v48 >= v60 )
        {
LABEL_53:
          v27 = (__int64)v47;
          goto LABEL_31;
        }
      }
      else
      {
        if ( (unsigned __int8)sub_AC47B0(28) )
          v49 = sub_AD5570(28, (__int64)v47, v50, 0, 0);
        else
          v49 = sub_AABE40(0x1Cu, v47, v50);
LABEL_45:
        if ( !v49 )
          goto LABEL_50;
        v47 = (unsigned __int8 *)v49;
LABEL_47:
        v46 = ++v48;
        if ( v48 >= v60 )
          goto LABEL_53;
      }
    }
    v49 = v51((__int64)v75, 28u, v47, v50);
    goto LABEL_45;
  }
  v27 = v67[0];
  if ( a4 )
  {
LABEL_30:
    v38 = a5;
    a5 = a6;
    a6 = v38;
  }
LABEL_31:
  v64 = 257;
  v39 = sub_BD2C40(72, 3u);
  v41 = (__int64)v39;
  if ( v39 )
    sub_B4C9A0((__int64)v39, a6, a5, v27, 3u, v40, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v76 + 2))(v76, v41, v63, v72, v73);
  v42 = (unsigned __int64)v68;
  v43 = &v68[16 * (unsigned int)v69];
  if ( v68 != v43 )
  {
    do
    {
      v44 = *(_QWORD *)(v42 + 8);
      v45 = *(_DWORD *)v42;
      v42 += 16LL;
      sub_B99FD0(v41, v45, v44);
    }
    while ( v43 != (_BYTE *)v42 );
  }
  if ( v65 != v67 )
    _libc_free((unsigned __int64)v65);
  nullsub_61();
  v83 = &unk_49DA100;
  nullsub_63();
  if ( v68 != v70 )
    _libc_free((unsigned __int64)v68);
}
