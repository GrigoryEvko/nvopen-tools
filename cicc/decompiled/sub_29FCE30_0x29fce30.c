// Function: sub_29FCE30
// Address: 0x29fce30
//
__int64 __fastcall sub_29FCE30(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        unsigned int a5,
        float a6,
        __m128i a7)
{
  unsigned int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r13
  unsigned __int64 v13; // rax
  int v14; // ecx
  _QWORD *v15; // rdx
  _DWORD *v16; // r15
  __int64 v17; // r14
  __int64 v18; // rdx
  __int64 v19; // r14
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned __int8 *v22; // rbx
  __int64 (__fastcall *v23)(void **, __int64, unsigned __int8 *, unsigned __int8 *, __int64); // rax
  __int64 v24; // r14
  unsigned __int64 v26; // r12
  _BYTE *v27; // rbx
  __int64 v28; // rdx
  unsigned int v29; // esi
  __int64 v30; // rdx
  _QWORD *v31; // r15
  _QWORD *v32; // r12
  __int64 v33; // rdx
  _QWORD *v34; // rdi
  _QWORD *v35; // rbx
  unsigned __int64 v36; // rsi
  __int64 v37; // [rsp-8h] [rbp-178h]
  void *v40; // [rsp+30h] [rbp-140h]
  unsigned __int8 *v42; // [rsp+40h] [rbp-130h]
  void *v44; // [rsp+50h] [rbp-120h] BYREF
  _QWORD *v45; // [rsp+58h] [rbp-118h]
  __int16 v46; // [rsp+70h] [rbp-100h]
  __int64 v47[4]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v48; // [rsp+A0h] [rbp-D0h]
  _BYTE *v49; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v50; // [rsp+B8h] [rbp-B8h]
  _BYTE v51[32]; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v52; // [rsp+E0h] [rbp-90h]
  __int64 v53; // [rsp+E8h] [rbp-88h]
  __int64 v54; // [rsp+F0h] [rbp-80h]
  __int64 *v55; // [rsp+F8h] [rbp-78h]
  void **v56; // [rsp+100h] [rbp-70h]
  void **v57; // [rsp+108h] [rbp-68h]
  __int64 v58; // [rsp+110h] [rbp-60h]
  int v59; // [rsp+118h] [rbp-58h]
  __int16 v60; // [rsp+11Ch] [rbp-54h]
  char v61; // [rsp+11Eh] [rbp-52h]
  __int64 v62; // [rsp+120h] [rbp-50h]
  __int64 v63; // [rsp+128h] [rbp-48h]
  void *v64; // [rsp+130h] [rbp-40h] BYREF
  void *v65; // [rsp+138h] [rbp-38h] BYREF

  v7 = _mm_cvtsi128_si32(a7);
  v58 = 0;
  v55 = (__int64 *)sub_BD5C60(a1);
  v56 = &v64;
  v57 = &v65;
  v49 = v51;
  v64 = &unk_49DA100;
  v50 = 0x200000000LL;
  v59 = 0;
  v65 = &unk_49DA0B0;
  v8 = *(_QWORD *)(a1 + 40);
  v60 = 512;
  v52 = v8;
  v61 = 7;
  v62 = 0;
  v63 = 0;
  v53 = a1 + 24;
  LOWORD(v54) = 0;
  v9 = *(_QWORD *)sub_B46C60(a1);
  v47[0] = v9;
  if ( v9 && (sub_B96E90((__int64)v47, v9, 1), (v12 = v47[0]) != 0) )
  {
    v13 = (unsigned __int64)v49;
    v14 = v50;
    v15 = &v49[16 * (unsigned int)v50];
    if ( v49 != (_BYTE *)v15 )
    {
      while ( *(_DWORD *)v13 )
      {
        v13 += 16LL;
        if ( v15 == (_QWORD *)v13 )
          goto LABEL_33;
      }
      *(_QWORD *)(v13 + 8) = v47[0];
      goto LABEL_8;
    }
LABEL_33:
    if ( (unsigned int)v50 >= (unsigned __int64)HIDWORD(v50) )
    {
      v36 = (unsigned int)v50 + 1LL;
      if ( HIDWORD(v50) < v36 )
      {
        sub_C8D5F0((__int64)&v49, v51, v36, 0x10u, v10, v11);
        v15 = &v49[16 * (unsigned int)v50];
      }
      *v15 = 0;
      v15[1] = v12;
      v12 = v47[0];
      LODWORD(v50) = v50 + 1;
    }
    else
    {
      if ( v15 )
      {
        *(_DWORD *)v15 = 0;
        v15[1] = v12;
        v14 = v50;
        v12 = v47[0];
      }
      LODWORD(v50) = v14 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v49, 0);
    v12 = v47[0];
  }
  if ( v12 )
LABEL_8:
    sub_B91220((__int64)v47, v12);
  v16 = sub_C33310();
  sub_C3B170((__int64)v47, _mm_cvtsi32_si128(v7));
  sub_C407B0(&v44, v47, v16);
  sub_C338F0((__int64)v47);
  v17 = sub_AC8EA0(v55, (__int64 *)&v44);
  v40 = sub_C33340();
  if ( v44 == v40 )
  {
    if ( v45 )
    {
      v33 = *(v45 - 1);
      v34 = &v45[3 * v33];
      if ( v45 != v34 )
      {
        v35 = &v45[3 * v33];
        do
        {
          v35 -= 3;
          sub_91D830(v35);
        }
        while ( v45 != v35 );
        v34 = v35;
      }
      j_j_j___libc_free_0_0((unsigned __int64)(v34 - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v44);
  }
  v18 = *(_QWORD *)(a4 + 8);
  if ( *(_BYTE *)(v18 + 8) != 2 )
    v17 = sub_AA93C0(0x2Eu, v17, v18);
  if ( (unsigned __int8)sub_B2D610(*(_QWORD *)(v52 + 72), 72) )
    LOBYTE(v60) = 1;
  HIDWORD(v44) = 0;
  v48 = 257;
  v42 = (unsigned __int8 *)sub_B35C90((__int64)&v49, a5, a4, v17, (__int64)v47, 0, (unsigned int)v44, 0);
  sub_C3B170((__int64)v47, (__m128i)LODWORD(a6));
  sub_C407B0(&v44, v47, v16);
  sub_C338F0((__int64)v47);
  v19 = sub_AC8EA0(v55, (__int64 *)&v44);
  if ( v40 == v44 )
  {
    if ( v45 )
    {
      v30 = *(v45 - 1);
      v31 = &v45[3 * v30];
      if ( v45 != v31 )
      {
        v32 = &v45[3 * v30];
        do
        {
          v32 -= 3;
          sub_91D830(v32);
        }
        while ( v45 != v32 );
        v31 = v32;
      }
      j_j_j___libc_free_0_0((unsigned __int64)(v31 - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v44);
  }
  v20 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(v20 + 8) != 2 )
    v19 = sub_AA93C0(0x2Eu, v19, v20);
  if ( (unsigned __int8)sub_B2D610(*(_QWORD *)(v52 + 72), 72) )
    LOBYTE(v60) = 1;
  v48 = 257;
  HIDWORD(v44) = 0;
  v21 = sub_B35C90((__int64)&v49, a3, a2, v19, (__int64)v47, 0, (unsigned int)v44, 0);
  v46 = 257;
  v22 = (unsigned __int8 *)v21;
  v23 = (__int64 (__fastcall *)(void **, __int64, unsigned __int8 *, unsigned __int8 *, __int64))*((_QWORD *)*v56 + 2);
  if ( (char *)v23 != (char *)sub_9202E0 )
  {
    v24 = v23(v56, 29, v22, v42, v37);
    goto LABEL_26;
  }
  if ( *v22 <= 0x15u && *v42 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(29) )
      v24 = sub_AD5570(29, (__int64)v22, v42, 0, 0);
    else
      v24 = sub_AABE40(0x1Du, v22, v42);
LABEL_26:
    if ( v24 )
      goto LABEL_27;
  }
  v48 = 257;
  v24 = sub_B504D0(29, (__int64)v22, (__int64)v42, (__int64)v47, 0, 0);
  (*((void (__fastcall **)(void **, __int64, void **, __int64, __int64))*v57 + 2))(v57, v24, &v44, v53, v54);
  v26 = (unsigned __int64)v49;
  v27 = &v49[16 * (unsigned int)v50];
  if ( v49 != v27 )
  {
    do
    {
      v28 = *(_QWORD *)(v26 + 8);
      v29 = *(_DWORD *)v26;
      v26 += 16LL;
      sub_B99FD0(v24, v29, v28);
    }
    while ( v27 != (_BYTE *)v26 );
  }
LABEL_27:
  nullsub_61();
  v64 = &unk_49DA100;
  nullsub_63();
  if ( v49 != v51 )
    _libc_free((unsigned __int64)v49);
  return v24;
}
