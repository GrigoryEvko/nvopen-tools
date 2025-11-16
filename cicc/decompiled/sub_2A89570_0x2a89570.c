// Function: sub_2A89570
// Address: 0x2a89570
//
__int64 __fastcall sub_2A89570(__int64 a1, unsigned int a2, _QWORD *a3, __int64 a4, _BYTE *a5)
{
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r13
  unsigned __int64 v13; // rsi
  unsigned int *v14; // rax
  int v15; // ecx
  unsigned int *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r13
  unsigned __int8 *v20; // r13
  __int64 v21; // r12
  __int64 **v23; // r15
  int v24; // ebx
  unsigned int v25; // esi
  int v26; // ebx
  _BYTE *v27; // rax
  __int64 v28; // rax
  unsigned __int8 *v29; // r15
  __int64 (__fastcall *v30)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  _BYTE *v31; // rax
  __int64 v32; // rax
  unsigned __int8 *v33; // r15
  __int64 (__fastcall *v34)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v35; // r14
  __int64 v36; // rax
  unsigned int *v37; // r14
  unsigned int *v38; // r15
  __int64 v39; // rdx
  unsigned int v40; // esi
  unsigned int *v41; // r13
  unsigned int *v42; // r15
  __int64 v43; // rdx
  unsigned int v44; // esi
  _QWORD *v47; // [rsp+38h] [rbp-138h]
  unsigned __int8 *v48; // [rsp+38h] [rbp-138h]
  unsigned __int64 v49; // [rsp+48h] [rbp-128h]
  unsigned __int64 v50; // [rsp+48h] [rbp-128h]
  int v51[8]; // [rsp+50h] [rbp-120h] BYREF
  __int16 v52; // [rsp+70h] [rbp-100h]
  __int64 v53; // [rsp+80h] [rbp-F0h] BYREF
  unsigned int v54; // [rsp+88h] [rbp-E8h]
  __int16 v55; // [rsp+A0h] [rbp-D0h]
  unsigned int *v56; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v57; // [rsp+B8h] [rbp-B8h]
  _BYTE v58[32]; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v59; // [rsp+E0h] [rbp-90h]
  __int64 v60; // [rsp+E8h] [rbp-88h]
  __int64 v61; // [rsp+F0h] [rbp-80h]
  __int64 v62; // [rsp+F8h] [rbp-78h]
  void **v63; // [rsp+100h] [rbp-70h]
  void **v64; // [rsp+108h] [rbp-68h]
  __int64 v65; // [rsp+110h] [rbp-60h]
  int v66; // [rsp+118h] [rbp-58h]
  __int16 v67; // [rsp+11Ch] [rbp-54h]
  char v68; // [rsp+11Eh] [rbp-52h]
  __int64 v69; // [rsp+120h] [rbp-50h]
  __int64 v70; // [rsp+128h] [rbp-48h]
  void *v71; // [rsp+130h] [rbp-40h] BYREF
  void *v72; // [rsp+138h] [rbp-38h] BYREF

  v47 = (_QWORD *)*a3;
  v49 = sub_9208B0((__int64)a5, (__int64)a3);
  v65 = 0;
  v62 = sub_BD5C60(a4);
  v63 = &v71;
  v64 = &v72;
  v56 = (unsigned int *)v58;
  v71 = &unk_49DA100;
  v57 = 0x200000000LL;
  v66 = 0;
  v72 = &unk_49DA0B0;
  v8 = *(_QWORD *)(a4 + 40);
  v67 = 512;
  v59 = v8;
  v60 = a4 + 24;
  v68 = 7;
  v69 = 0;
  v70 = 0;
  LOWORD(v61) = 0;
  v9 = *(_QWORD *)sub_B46C60(a4);
  v53 = v9;
  if ( !v9 || (sub_B96E90((__int64)&v53, v9, 1), (v12 = v53) == 0) )
  {
    v13 = 0;
    sub_93FB40((__int64)&v56, 0);
    v12 = v53;
    goto LABEL_50;
  }
  v13 = (unsigned int)v57;
  v14 = v56;
  v15 = v57;
  v16 = &v56[4 * (unsigned int)v57];
  if ( v56 == v16 )
  {
LABEL_52:
    if ( (unsigned int)v57 >= (unsigned __int64)HIDWORD(v57) )
    {
      v13 = (unsigned int)v57 + 1LL;
      if ( HIDWORD(v57) < v13 )
      {
        v13 = (unsigned __int64)v58;
        sub_C8D5F0((__int64)&v56, v58, (unsigned int)v57 + 1LL, 0x10u, v10, v11);
        v16 = &v56[4 * (unsigned int)v57];
      }
      *(_QWORD *)v16 = 0;
      *((_QWORD *)v16 + 1) = v12;
      v12 = v53;
      LODWORD(v57) = v57 + 1;
    }
    else
    {
      if ( v16 )
      {
        *v16 = 0;
        *((_QWORD *)v16 + 1) = v12;
        v15 = v57;
        v12 = v53;
      }
      LODWORD(v57) = v15 + 1;
    }
LABEL_50:
    if ( !v12 )
      goto LABEL_9;
    goto LABEL_8;
  }
  while ( *v14 )
  {
    v14 += 4;
    if ( v16 == v14 )
      goto LABEL_52;
  }
  *((_QWORD *)v14 + 1) = v53;
LABEL_8:
  v13 = v12;
  sub_B91220((__int64)&v53, v12);
LABEL_9:
  v17 = *(_QWORD *)(a1 - 32);
  if ( !v17 || *(_BYTE *)v17 || *(_QWORD *)(v17 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  v18 = 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  v19 = *(_QWORD *)(a1 + v18);
  if ( ((*(_DWORD *)(v17 + 36) - 243) & 0xFFFFFFFD) == 0 )
  {
    if ( v49 >> 3 == 1 )
    {
LABEL_40:
      v36 = sub_B43CB0(a4);
      v21 = (__int64)sub_2A88A40((_BYTE *)v19, (__int64)a3, (__int64 *)&v56, v36);
      goto LABEL_18;
    }
    v55 = 257;
    v23 = (__int64 **)sub_BCCE00(v47, 8 * (unsigned int)(v49 >> 3));
    v24 = sub_BCB060(*(_QWORD *)(v19 + 8));
    v25 = 49;
    if ( v24 != (unsigned int)sub_BCB060((__int64)v23) )
      v25 = 39;
    v26 = 1;
    v50 = v49 >> 3;
    v48 = (unsigned __int8 *)sub_2A882B0((__int64 *)&v56, v25, v19, v23, (__int64)&v53, 0, v51[0], 0);
    v19 = (__int64)v48;
    while ( 1 )
    {
      while ( 1 )
      {
        v55 = 257;
        if ( (unsigned int)(2 * v26) > v50 )
          break;
        v31 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v19 + 8), (unsigned int)(8 * v26), 0);
        v32 = sub_920A70(&v56, (_BYTE *)v19, v31, (__int64)&v53, 0, 0);
        v52 = 257;
        v33 = (unsigned __int8 *)v32;
        v34 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, unsigned __int8 *))*((_QWORD *)*v63 + 2);
        if ( v34 == sub_9202E0 )
        {
          if ( *(_BYTE *)v19 > 0x15u || *v33 > 0x15u )
          {
LABEL_45:
            v55 = 257;
            v35 = sub_B504D0(29, v19, (__int64)v33, (__int64)&v53, 0, 0);
            (*((void (__fastcall **)(void **, __int64, int *, __int64, __int64))*v64 + 2))(v64, v35, v51, v60, v61);
            v41 = v56;
            v42 = &v56[4 * (unsigned int)v57];
            if ( v56 != v42 )
            {
              do
              {
                v43 = *((_QWORD *)v41 + 1);
                v44 = *v41;
                v41 += 4;
                sub_B99FD0(v35, v44, v43);
              }
              while ( v42 != v41 );
            }
            goto LABEL_39;
          }
          if ( (unsigned __int8)sub_AC47B0(29) )
            v35 = sub_AD5570(29, v19, v33, 0, 0);
          else
            v35 = sub_AABE40(0x1Du, (unsigned __int8 *)v19, v33);
        }
        else
        {
          v35 = v34((__int64)v63, 29u, (_BYTE *)v19, v33);
        }
        if ( !v35 )
          goto LABEL_45;
LABEL_39:
        v26 *= 2;
        v19 = v35;
        if ( v26 == v50 )
          goto LABEL_40;
      }
      v27 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v19 + 8), 8, 0);
      v28 = sub_920A70(&v56, (_BYTE *)v19, v27, (__int64)&v53, 0, 0);
      v52 = 257;
      v29 = (unsigned __int8 *)v28;
      v30 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v63 + 2);
      if ( v30 == sub_9202E0 )
      {
        if ( *v48 > 0x15u || *v29 > 0x15u )
        {
LABEL_41:
          v55 = 257;
          v19 = sub_B504D0(29, (__int64)v48, (__int64)v29, (__int64)&v53, 0, 0);
          (*((void (__fastcall **)(void **, __int64, int *, __int64, __int64))*v64 + 2))(v64, v19, v51, v60, v61);
          v37 = v56;
          v38 = &v56[4 * (unsigned int)v57];
          if ( v56 != v38 )
          {
            do
            {
              v39 = *((_QWORD *)v37 + 1);
              v40 = *v37;
              v37 += 4;
              sub_B99FD0(v19, v40, v39);
            }
            while ( v38 != v37 );
          }
          goto LABEL_31;
        }
        if ( (unsigned __int8)sub_AC47B0(29) )
          v19 = sub_AD5570(29, (__int64)v48, v29, 0, 0);
        else
          v19 = sub_AABE40(0x1Du, v48, v29);
      }
      else
      {
        v19 = v30((__int64)v63, 29u, v48, v29);
      }
      if ( !v19 )
        goto LABEL_41;
LABEL_31:
      if ( ++v26 == v50 )
        goto LABEL_40;
    }
  }
  v20 = sub_BD3990(*(unsigned __int8 **)(a1 + v18), v13);
  v54 = sub_AE43F0((__int64)a5, *((_QWORD *)v20 + 1));
  if ( v54 > 0x40 )
    sub_C43690((__int64)&v53, a2, 0);
  else
    v53 = a2;
  v21 = sub_971820((__int64)v20, (__int64)a3, (__int64)&v53, a5);
  if ( v54 > 0x40 && v53 )
    j_j___libc_free_0_0(v53);
LABEL_18:
  nullsub_61();
  v71 = &unk_49DA100;
  nullsub_63();
  if ( v56 != (unsigned int *)v58 )
    _libc_free((unsigned __int64)v56);
  return v21;
}
