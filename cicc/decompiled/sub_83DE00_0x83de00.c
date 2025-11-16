// Function: sub_83DE00
// Address: 0x83de00
//
__int64 __fastcall sub_83DE00(
        const __m128i *a1,
        __int64 a2,
        __int64 a3,
        int a4,
        __int64 a5,
        _DWORD *a6,
        _DWORD *a7,
        __int64 *a8,
        _DWORD *a9)
{
  const __m128i *v9; // r15
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 *v14; // r9
  __int64 v15; // r14
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rdx
  const __m128i *i; // rax
  __int64 v21; // rbx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // r13
  unsigned int v25; // r13d
  char v26; // al
  __int64 v27; // rdx
  __int64 v28; // r12
  bool v29; // si
  __m128i *v30; // r12
  _QWORD *v31; // r13
  __int64 v32; // r12
  _QWORD *v33; // rbx
  __int64 v34; // rcx
  __int64 v35; // rbx
  __int64 v36; // r8
  __int64 *v37; // r9
  __int64 v38; // rdx
  __int64 v39; // r13
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 *v43; // r9
  char v45; // r15
  char v46; // al
  int v47; // r13d
  int v48; // r12d
  __int64 v49; // rdi
  __int64 v50; // rdi
  __int64 v52; // rax
  __int64 *v53; // rax
  __int64 v54; // r14
  unsigned int v55; // [rsp+4h] [rbp-CCh]
  unsigned int v56; // [rsp+8h] [rbp-C8h]
  __int64 v57; // [rsp+10h] [rbp-C0h]
  __int64 v58; // [rsp+18h] [rbp-B8h]
  int v60; // [rsp+40h] [rbp-90h]
  unsigned int v61; // [rsp+44h] [rbp-8Ch]
  int v64; // [rsp+54h] [rbp-7Ch]
  int v65; // [rsp+58h] [rbp-78h]
  int v66; // [rsp+5Ch] [rbp-74h]
  int v67; // [rsp+64h] [rbp-6Ch] BYREF
  __int64 *v68; // [rsp+68h] [rbp-68h] BYREF
  _QWORD *v69; // [rsp+70h] [rbp-60h] BYREF
  __int64 v70; // [rsp+78h] [rbp-58h] BYREF
  _BYTE v71[80]; // [rsp+80h] [rbp-50h] BYREF

  v9 = a1;
  v64 = a2;
  v65 = a3;
  v10 = sub_82BD70(a1, a2, a3);
  v15 = *(_QWORD *)(v10 + 1024);
  v16 = v10;
  if ( v15 == *(_QWORD *)(v10 + 1016) )
    sub_8332F0(v10, a2, v11, v12, v13, v14);
  v17 = *(_QWORD *)(v16 + 1008) + 40 * v15;
  if ( v17 )
  {
    *(_BYTE *)v17 &= 0xFCu;
    *(_QWORD *)(v17 + 8) = 0;
    *(_QWORD *)(v17 + 16) = 0;
    *(_QWORD *)(v17 + 24) = 0;
    *(_QWORD *)(v17 + 32) = 0;
  }
  *(_QWORD *)(v16 + 1024) = v15 + 1;
  if ( a7 )
    *a7 = 0;
  *a9 = 0;
  *a6 = 0;
  if ( a1[8].m128i_i8[12] == 12 )
  {
    do
      v9 = (const __m128i *)v9[10].m128i_i64[0];
    while ( v9[8].m128i_i8[12] == 12 );
  }
  v18 = (__int64)v9;
  if ( (unsigned int)sub_8D23B0(v9) )
  {
    v18 = (__int64)v9;
    if ( (unsigned int)sub_8D3A70(v9) )
    {
      a2 = 0;
      v18 = (__int64)v9;
      sub_8AD220(v9, 0);
    }
  }
  for ( i = v9; i[8].m128i_i8[12] == 12; i = (const __m128i *)i[10].m128i_i64[0] )
    ;
  v58 = *(_QWORD *)(i->m128i_i64[0] + 96);
  v21 = *(_QWORD *)(v58 + 8);
  if ( (*(_BYTE *)(v58 + 177) & 0x41) != 0x40 && (v9[11].m128i_i8[1] & 0x20) == 0 )
  {
    if ( !v21 )
      goto LABEL_51;
    v69 = 0;
    v60 = 2;
    v56 = -1;
    v61 = 0;
    v66 = 0;
    v55 = 0;
    v57 = 0;
    while ( 1 )
    {
      v24 = sub_82C1B0(v21, (__int64)&v69, (__int64)a8, (__int64)v71);
      if ( v24 )
        break;
LABEL_38:
      if ( v60 != 1 )
      {
        v23 = v61;
        v66 = 1;
        v60 = 1;
        if ( !v61 )
          continue;
        v22 = v56;
        if ( v56 )
          continue;
      }
      a2 = a5;
      sub_82D8D0((__int64 *)&v69, a5, &v70, a6, v22, v23);
      v18 = (unsigned int)v70;
      v31 = v69;
      if ( (_DWORD)v70 )
      {
        v32 = 0;
        if ( v69 )
          goto LABEL_43;
LABEL_94:
        a2 = v55;
        LOBYTE(v19) = v57 != 0;
        if ( v55 != 0 || v57 == 0 || !a7 )
          goto LABEL_51;
        v32 = v57;
        *a7 = 1;
        goto LABEL_52;
      }
      if ( *a6 )
      {
        if ( !v69 )
          goto LABEL_94;
        v32 = 0;
      }
      else
      {
        if ( !v69 )
          goto LABEL_94;
        v32 = v69[1];
      }
      do
      {
LABEL_43:
        v33 = v31;
        v31 = (_QWORD *)*v31;
        sub_725130((__int64 *)v33[5]);
        v18 = v33[15];
        sub_82D8A0((_QWORD *)v18);
        *v33 = qword_4D03C68;
        qword_4D03C68 = v33;
      }
      while ( v31 );
      if ( v32 )
      {
        v18 = *(_QWORD *)(v32 + 88);
        if ( (*(_BYTE *)(v18 + 194) & 4) != 0
          && (*(_BYTE *)(v18 + 206) & 0x10) == 0
          && (*(_BYTE *)(v18 + 193) & 4) != 0
          && (*(_BYTE *)(v58 + 177) & 0x40) != 0
          && !sub_72F570(v18) )
        {
          *a9 = 1;
          goto LABEL_51;
        }
        goto LABEL_52;
      }
      goto LABEL_94;
    }
    while ( 1 )
    {
      v26 = *(_BYTE *)(v24 + 80);
      if ( (v26 == 20) != v66 )
        goto LABEL_20;
      v27 = *(_QWORD *)(v24 + 88);
      if ( v26 == 20 )
        v27 = *(_QWORD *)(v27 + 176);
      v28 = **(_QWORD **)(*(_QWORD *)(v27 + 152) + 168LL);
      if ( a4 )
      {
        if ( *(char *)(v27 + 193) < 0 )
          goto LABEL_20;
      }
      v29 = v28 != 0;
      if ( (*(_BYTE *)(v27 + 206) & 0x18) == 0x18 )
      {
        if ( v28 )
        {
          v29 = 1;
          if ( (unsigned int)sub_8D3110(*(_QWORD *)(v28 + 8)) )
            goto LABEL_20;
        }
      }
      if ( v61
        && *(_BYTE *)(v24 + 80) == 20
        && v29
        && ((unsigned int)sub_8D3070(*(_QWORD *)(v28 + 8)) || !v65 && (unsigned int)sub_8D3110(*(_QWORD *)(v28 + 8))) )
      {
        v50 = sub_8D46C0(*(_QWORD *)(v28 + 8));
        if ( !((*(_BYTE *)(v50 + 140) & 0xFB) == 8 ? v56 & ~(unsigned int)sub_8D4C10(v50, dword_4F077C4 != 2) : v56) )
          goto LABEL_20;
      }
      v30 = (__m128i *)qword_4D03C60;
      if ( qword_4D03C60 )
        qword_4D03C60 = (_QWORD *)*qword_4D03C60;
      else
        v30 = (__m128i *)sub_823970(104);
      sub_82D850((__int64)v30);
      sub_83DAC0(v24, v9, v64, v65, v30, (__int64 *)&v68, &v70, &v67);
      if ( v30->m128i_i32[2] == 7 )
      {
        if ( v67 )
        {
          if ( v57 )
            v55 = 1;
          else
            v57 = v24;
        }
        sub_82D8A0(v30);
        sub_725130(v68);
        v24 = sub_82C230(v71);
        if ( !v24 )
          goto LABEL_38;
      }
      else
      {
        if ( *(_BYTE *)(v24 + 80) == 20 )
        {
          sub_82B9D0(v24, v21, v70, 0, (__int64)v68, (__int64)v30, (__int64 *)&v69);
        }
        else
        {
          sub_82B8E0(v24, v21, (__int64)v30, (__int64 *)&v69);
          v25 = v30->m128i_u32[2];
          if ( !v25 && (unsigned int)sub_8D2FB0(v30[2].m128i_i64[0]) )
          {
            v49 = sub_8D46C0(v30[2].m128i_i64[0]);
            if ( (*(_BYTE *)(v49 + 140) & 0xFB) == 8 )
              v25 = sub_8D4C10(v49, dword_4F077C4 != 2) & v56 & ~v64;
            v56 = v25;
            v61 = 1;
          }
        }
LABEL_20:
        v24 = sub_82C230(v71);
        if ( !v24 )
          goto LABEL_38;
      }
    }
  }
  if ( (v64 & 0xFFFFFFFE) != 0 )
    goto LABEL_51;
  *a9 = 1;
  if ( v65 == 0 && a8 == 0 || !v21 || (v9[11].m128i_i8[1] & 0x20) != 0 || (unsigned __int8)(v9[8].m128i_i8[12] - 9) > 2u )
    goto LABEL_51;
  v45 = *(_BYTE *)(v21 + 80);
  v46 = v45;
  if ( v45 == 17 )
  {
    v21 = *(_QWORD *)(v21 + 88);
    if ( !v21 )
    {
LABEL_64:
      v32 = 0;
      *a9 = 0;
      goto LABEL_52;
    }
    v46 = *(_BYTE *)(v21 + 80);
  }
  v47 = 0;
  v48 = 0;
  if ( v46 == 10 )
    goto LABEL_87;
  while ( v45 == 17 )
  {
    v21 = *(_QWORD *)(v21 + 8);
    if ( !v21 )
      break;
    if ( *(_BYTE *)(v21 + 80) == 10 )
    {
LABEL_87:
      v52 = *(_QWORD *)(v21 + 88);
      if ( (*(_BYTE *)(v52 + 194) & 4) != 0 )
      {
        v53 = *(__int64 **)(*(_QWORD *)(v52 + 152) + 168LL);
        v54 = *v53;
        v18 = *(_QWORD *)(*v53 + 8);
        if ( (unsigned int)sub_8D3110(v18) == v65 )
        {
          v48 = 1;
LABEL_100:
          v47 = 1;
          if ( a8 )
          {
            a2 = 1;
            v18 = v21;
            if ( !(unsigned int)sub_884000(v21, 1) )
              *a8 = v21;
          }
        }
        else if ( !v48 && v65 )
        {
          v18 = *(_QWORD *)(v54 + 8);
          if ( (unsigned int)sub_8D4D20(v18) )
            goto LABEL_100;
          v48 = 0;
        }
      }
    }
  }
  if ( !v47 )
    goto LABEL_64;
LABEL_51:
  v32 = 0;
LABEL_52:
  v35 = sub_82BD70(v18, a2, v19);
  v38 = *(_QWORD *)(v35 + 1008);
  v39 = *(_QWORD *)(v38 + 8 * (5LL * *(_QWORD *)(v35 + 1024) - 5) + 32);
  if ( v39 )
  {
    sub_823A00(*(_QWORD *)v39, 16LL * (unsigned int)(*(_DWORD *)(v39 + 8) + 1), v38, v34, v36, v37);
    sub_823A00(v39, 16, v40, v41, v42, v43);
  }
  --*(_QWORD *)(v35 + 1024);
  return v32;
}
