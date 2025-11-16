// Function: sub_34C6AF0
// Address: 0x34c6af0
//
__int64 __fastcall sub_34C6AF0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, char a6)
{
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // r12
  int v13; // ecx
  __int64 v14; // rdi
  __int64 (*v15)(); // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  int v21; // ebx
  int v22; // eax
  __int64 v23; // r13
  __int64 v24; // r12
  __int64 v25; // rbx
  __int64 v26; // r9
  _QWORD *v27; // r11
  _QWORD *v28; // r14
  __int64 i; // rdx
  __int64 v30; // rax
  __int64 j; // r9
  __int64 *v32; // rdi
  __int64 k; // rcx
  _QWORD *v34; // rax
  __int64 v36; // r13
  int v37; // r12d
  __int64 v38; // rsi
  __int64 v39; // r13
  char v40; // r12
  char v41; // al
  size_t v42; // rdx
  _QWORD *v43; // r13
  _QWORD *v44; // rbx
  unsigned __int64 v45; // rdx
  _QWORD *v46; // rax
  _QWORD *v47; // rdi
  __int64 v48; // rsi
  __int64 v49; // rcx
  __int64 v50; // rcx
  _QWORD *v51; // r8
  __int64 v52; // rax
  _QWORD *v53; // rdi
  __int64 v54; // rsi
  __int64 v55; // rdi
  __int64 (*v56)(); // rdx
  int v57; // eax
  unsigned int v58; // [rsp+10h] [rbp-90h]
  char v59; // [rsp+14h] [rbp-8Ch]
  unsigned __int8 v60; // [rsp+14h] [rbp-8Ch]
  __int64 v61; // [rsp+18h] [rbp-88h]
  _QWORD *v62; // [rsp+18h] [rbp-88h]
  __int64 v63; // [rsp+18h] [rbp-88h]
  void *p_s; // [rsp+20h] [rbp-80h] BYREF
  __int64 v65; // [rsp+28h] [rbp-78h]
  __int64 s; // [rsp+30h] [rbp-70h] BYREF
  int v67; // [rsp+38h] [rbp-68h]
  unsigned int v68; // [rsp+60h] [rbp-40h]

  ++*(_QWORD *)(a1 + 24);
  if ( !*(_BYTE *)(a1 + 52) )
  {
    v59 = a6;
    v10 = 4 * (*(_DWORD *)(a1 + 44) - *(_DWORD *)(a1 + 48));
    v11 = *(unsigned int *)(a1 + 40);
    v61 = a5;
    if ( v10 < 0x20 )
      v10 = 32;
    if ( (unsigned int)v11 > v10 )
    {
      sub_C8C990(a1 + 24, (__int64)a2);
      a5 = v61;
      a6 = v59;
      goto LABEL_7;
    }
    memset(*(void **)(a1 + 32), -1, 8 * v11);
    a6 = v59;
    a5 = v61;
  }
  *(_QWORD *)(a1 + 44) = 0;
LABEL_7:
  v12 = a2[4];
  *(_BYTE *)(a1 + 128) = a6;
  *(_QWORD *)(a1 + 136) = a3;
  v13 = *(_DWORD *)(a1 + 132);
  *(_QWORD *)(a1 + 152) = a4;
  *(_QWORD *)(a1 + 160) = a5;
  *(_QWORD *)(a1 + 144) = v12;
  v58 = *(_DWORD *)(a2[1] + 544LL) - 42;
  if ( !v13 )
  {
    v43 = sub_C52410();
    v44 = v43 + 1;
    v45 = sub_C959E0();
    v46 = (_QWORD *)v43[2];
    if ( v46 )
    {
      v47 = v43 + 1;
      do
      {
        while ( 1 )
        {
          v48 = v46[2];
          v49 = v46[3];
          if ( v45 <= v46[4] )
            break;
          v46 = (_QWORD *)v46[3];
          if ( !v49 )
            goto LABEL_65;
        }
        v47 = v46;
        v46 = (_QWORD *)v46[2];
      }
      while ( v48 );
LABEL_65:
      if ( v44 != v47 && v45 >= v47[4] )
        v44 = v47;
    }
    if ( v44 == (_QWORD *)((char *)sub_C52410() + 8) )
      goto LABEL_76;
    v52 = v44[7];
    v51 = v44 + 6;
    if ( !v52 )
      goto LABEL_76;
    v53 = v44 + 6;
    do
    {
      while ( 1 )
      {
        v54 = *(_QWORD *)(v52 + 16);
        v50 = *(_QWORD *)(v52 + 24);
        if ( *(_DWORD *)(v52 + 32) >= dword_503AAC8 )
          break;
        v52 = *(_QWORD *)(v52 + 24);
        if ( !v50 )
          goto LABEL_74;
      }
      v53 = (_QWORD *)v52;
      v52 = *(_QWORD *)(v52 + 16);
    }
    while ( v54 );
LABEL_74:
    if ( v53 == v51 || dword_503AAC8 < *((_DWORD *)v53 + 8) || (v57 = qword_503AB48, *((int *)v53 + 9) <= 0) )
    {
LABEL_76:
      v55 = *(_QWORD *)(a1 + 136);
      v56 = *(__int64 (**)())(*(_QWORD *)v55 + 1496LL);
      v57 = 3;
      if ( v56 != sub_2FDC810 )
        v57 = ((__int64 (__fastcall *)(__int64, _QWORD *, __int64 (*)(), __int64, _QWORD *))v56)(v55, a2, v56, v50, v51);
    }
    *(_DWORD *)(a1 + 132) = v57;
  }
  if ( (*(_BYTE *)(*(_QWORD *)v12 + 344LL) & 4) != 0
    && ((v14 = *(_QWORD *)(a1 + 152), v15 = *(__int64 (**)())(*(_QWORD *)v14 + 528LL), v15 == sub_2FF52D0)
     || ((unsigned __int8 (__fastcall *)(__int64, _QWORD *))v15)(v14, a2)) )
  {
    *(_BYTE *)(a1 + 131) = 1;
  }
  else
  {
    *(_BYTE *)(a1 + 131) = 0;
    *(_QWORD *)(*(_QWORD *)v12 + 344LL) &= ~4uLL;
  }
  sub_34BA1B0((__int64)&p_s, (__int64)a2);
  sub_C7D6A0(*(_QWORD *)(a1 + 80), 16LL * *(unsigned int *)(a1 + 96), 8);
  v16 = v65;
  ++*(_QWORD *)(a1 + 72);
  p_s = (char *)p_s + 1;
  *(_QWORD *)(a1 + 80) = v16;
  v65 = 0;
  *(_QWORD *)(a1 + 88) = s;
  s = 0;
  *(_DWORD *)(a1 + 96) = v67;
  v67 = 0;
  sub_C7D6A0(0, 0, 8);
  v60 = 0;
  v62 = a2 + 40;
  while ( 1 )
  {
    v22 = sub_34C4890(a1, (__int64)a2, v17, v18, v19, v20);
    LOBYTE(v21) = v22 | *(_BYTE *)(a1 + 128) ^ 1;
    if ( (_BYTE)v21 )
    {
      v21 = sub_34C2D70(a1, (__int64)a2) | v22;
      if ( v58 > 1 )
        goto LABEL_16;
      goto LABEL_15;
    }
    if ( v58 > 1 )
      break;
LABEL_15:
    if ( (_BYTE)qword_503AA68 )
    {
      v39 = a2[41];
      if ( (_QWORD *)v39 != v62 )
      {
        v40 = 0;
        do
        {
          v41 = sub_34C0690(a1, v39);
          v39 = *(_QWORD *)(v39 + 8);
          v40 |= v41;
        }
        while ( v62 != (_QWORD *)v39 );
        LOBYTE(v21) = v40 | v21;
      }
    }
LABEL_16:
    if ( *(_BYTE *)(a1 + 130) )
    {
      v36 = a2[41];
      if ( v62 != (_QWORD *)v36 )
        goto LABEL_49;
    }
LABEL_17:
    if ( !(_BYTE)v21 )
      goto LABEL_22;
    v60 = v21;
  }
  if ( *(_BYTE *)(a1 + 130) )
  {
    v36 = a2[41];
    if ( v62 != (_QWORD *)v36 )
    {
LABEL_49:
      v37 = 0;
      do
      {
        v38 = v36;
        v36 = *(_QWORD *)(v36 + 8);
        v37 |= sub_34C56D0(a1, v38);
      }
      while ( (_QWORD *)v36 != v62 );
      LOBYTE(v21) = v37 | v21;
      goto LABEL_17;
    }
  }
LABEL_22:
  v23 = a2[8];
  if ( v23 )
  {
    v24 = (__int64)(*(_QWORD *)(v23 + 16) - *(_QWORD *)(v23 + 8)) >> 5;
    p_s = &s;
    v65 = 0x600000000LL;
    v25 = (unsigned int)v24;
    v26 = (unsigned int)(v24 + 63) >> 6;
    if ( (unsigned int)v26 > 6 )
    {
      v63 = (unsigned int)v26;
      sub_C8D5F0((__int64)&p_s, &s, (unsigned int)v26, 8u, v19, v26);
      memset(p_s, 0, 8 * v63);
      LODWORD(v65) = (unsigned int)(v24 + 63) >> 6;
    }
    else
    {
      if ( (_DWORD)v26 )
      {
        v42 = 8LL * (unsigned int)v26;
        if ( v42 )
        {
          memset(&s, 0, v42);
          LODWORD(v26) = (unsigned int)(v24 + 63) >> 6;
        }
      }
      LODWORD(v65) = v26;
    }
    v27 = (_QWORD *)a2[41];
    v28 = a2 + 40;
    v68 = v24;
    if ( v27 != v28 )
    {
      do
      {
        for ( i = v27[7]; (_QWORD *)i != v27 + 6; i = *(_QWORD *)(i + 8) )
        {
          v30 = *(_QWORD *)(i + 32);
          for ( j = v30 + 40LL * (*(_DWORD *)(i + 40) & 0xFFFFFF); j != v30; v30 += 40 )
          {
            if ( *(_BYTE *)v30 == 8 )
              *((_QWORD *)p_s + (*(_DWORD *)(v30 + 24) >> 6)) |= 1LL << *(_DWORD *)(v30 + 24);
          }
          if ( (*(_BYTE *)i & 4) == 0 )
          {
            while ( (*(_BYTE *)(i + 44) & 8) != 0 )
              i = *(_QWORD *)(i + 8);
          }
        }
        v27 = (_QWORD *)v27[1];
      }
      while ( v28 != v27 );
      v25 = v68;
    }
    v32 = (__int64 *)p_s;
    if ( (_DWORD)v25 )
    {
      for ( k = 0; k != v25; ++k )
      {
        if ( (v32[(unsigned int)k >> 6] & (1LL << k)) == 0 )
        {
          v60 = 1;
          v34 = (_QWORD *)(*(_QWORD *)(v23 + 8) + 32 * k);
          if ( *v34 != v34[1] )
            v34[1] = *v34;
        }
      }
    }
    if ( v32 != &s )
      _libc_free((unsigned __int64)v32);
  }
  return v60;
}
