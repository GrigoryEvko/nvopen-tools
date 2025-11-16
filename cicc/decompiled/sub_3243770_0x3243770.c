// Function: sub_3243770
// Address: 0x3243770
//
__int64 __fastcall sub_3243770(__int64 a1, _QWORD *a2, unsigned __int64 **a3, unsigned int a4)
{
  unsigned int v7; // ecx
  unsigned __int64 *v8; // rax
  unsigned __int64 *v9; // rbx
  __int16 v11; // ax
  __int16 v12; // dx
  char v13; // al
  __int64 v14; // rdx
  char v15; // r13
  unsigned __int64 v16; // rax
  int v17; // ecx
  unsigned __int64 *v18; // rax
  bool v19; // zf
  _QWORD *v20; // rcx
  _QWORD *v21; // r15
  int v22; // r13d
  _QWORD *v23; // rdi
  __int64 (__fastcall *v24)(__int64, unsigned int); // rax
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned int v27; // esi
  unsigned __int64 v28; // rbx
  __int64 v29; // rsi
  unsigned __int64 *v30; // r15
  int v31; // eax
  unsigned __int64 *v32; // rdx
  unsigned __int64 *v33; // rax
  unsigned __int64 *v34; // r15
  unsigned __int64 v35; // rbx
  int v36; // eax
  unsigned __int64 *v37; // rax
  unsigned __int64 v38; // rax
  int v39; // ecx
  unsigned __int64 *v40; // r15
  unsigned __int64 *v41; // r15
  int v42; // eax
  unsigned __int8 v43; // [rsp+8h] [rbp-C8h]
  __int64 v44; // [rsp+10h] [rbp-C0h]
  unsigned __int64 *v45; // [rsp+10h] [rbp-C0h]
  unsigned __int8 v46; // [rsp+1Fh] [rbp-B1h]
  unsigned __int8 v47; // [rsp+20h] [rbp-B0h]
  _QWORD *v48; // [rsp+20h] [rbp-B0h]
  int v49; // [rsp+20h] [rbp-B0h]
  int v50; // [rsp+20h] [rbp-B0h]
  int v51; // [rsp+20h] [rbp-B0h]
  __int64 v52; // [rsp+28h] [rbp-A8h]
  char v53; // [rsp+28h] [rbp-A8h]
  unsigned __int64 *v54; // [rsp+30h] [rbp-A0h]
  __m128i v55; // [rsp+40h] [rbp-90h] BYREF
  __m128i v56; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v57[4]; // [rsp+60h] [rbp-70h] BYREF
  char v58; // [rsp+70h] [rbp-60h]
  __m128i v59; // [rsp+80h] [rbp-50h] BYREF
  char v60; // [rsp+90h] [rbp-40h]

  sub_AF47B0((__int64)v57, *a3, a3[1]);
  v7 = -2;
  if ( v58 )
    v7 = v57[0];
  v46 = sub_3242390(a1, a2, a4, v7);
  if ( !v46 )
  {
    *(_BYTE *)(a1 + 100) &= 0xF8u;
    return v46;
  }
  v8 = *a3;
  if ( a3[1] == *a3 )
  {
    v47 = 0;
    v9 = v54;
  }
  else
  {
    v59.m128i_i64[0] = (__int64)*a3;
    v59.m128i_i8[8] = 1;
    v9 = v8;
    if ( *v8 != 4096 )
    {
      if ( *(_DWORD *)(a1 + 32) > 1u )
      {
        if ( !*(_BYTE *)(a1 + 8) )
          goto LABEL_8;
LABEL_55:
        sub_3243700((_BYTE *)a1);
        goto LABEL_8;
      }
      v47 = v46;
      v43 = v46;
      v12 = (*(_WORD *)(a1 + 100) >> 6) & 7;
      goto LABEL_16;
    }
    v47 = v46;
  }
  if ( *(_BYTE *)(a1 + 8) && *(_DWORD *)(a1 + 32) > 1u )
    goto LABEL_55;
  v11 = *(_WORD *)(a1 + 100) >> 6;
  LOBYTE(v12) = v11 & 7;
  if ( (v11 & 4) != 0 )
  {
    v43 = 0;
  }
  else
  {
    v43 = 0;
    if ( (*(_BYTE *)(a1 + 100) & 7) != 2 )
      goto LABEL_31;
  }
LABEL_16:
  if ( (v12 & 1) != 0 )
  {
LABEL_31:
    sub_AF47B0((__int64)&v59, *a3, a3[1]);
    v20 = *(_QWORD **)(a1 + 24);
    v48 = &v20[3 * *(unsigned int *)(a1 + 32)];
    if ( v20 == v48 )
    {
LABEL_44:
      if ( (*(_BYTE *)(a1 + 100) & 0x40) != 0 )
      {
        sub_3243680(a1);
        if ( ((*(_WORD *)(a1 + 100) >> 6) & 6) == 0
          && !v43
          && (sub_32420F0(a1) || ((*(_BYTE *)(a1 + 101) >> 1) & 0xFu) > 3) )
        {
          (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 8LL))(a1, 159, 0);
        }
      }
      *(_DWORD *)(a1 + 32) = 0;
      v18 = *a3;
      if ( a3[1] != *a3 )
      {
        v55.m128i_i64[0] = (__int64)*a3;
        v19 = *(_WORD *)(a1 + 96) == 0;
        v55.m128i_i8[8] = 1;
        v56 = _mm_loadu_si128(&v55);
        if ( !v19 )
          goto LABEL_47;
      }
      return v46;
    }
    v21 = *(_QWORD **)(a1 + 24);
    v22 = 0;
    v53 = v60;
    while ( 1 )
    {
      v28 = (unsigned int)(*((_DWORD *)v21 + 2) + v22);
      v22 += *((_DWORD *)v21 + 2);
      if ( sub_32420F0(a1) )
        break;
      v29 = *v21;
      if ( (__int64)*v21 >= 0 )
        goto LABEL_41;
LABEL_36:
      if ( !v53 || v28 <= v59.m128i_i64[0] )
      {
        v27 = *((_DWORD *)v21 + 2);
        v21 += 3;
        sub_32422A0((_QWORD *)a1, v27, 0);
        if ( v48 != v21 )
          continue;
      }
      goto LABEL_44;
    }
    v23 = *(_QWORD **)(*(_QWORD *)(a1 + 16) + 184LL);
    v24 = *(__int64 (__fastcall **)(__int64, unsigned int))(*v23 + 480LL);
    if ( v24 == sub_31D48B0 )
    {
      v44 = *v21;
      v25 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v23[29] + 16LL) + 200LL))(*(_QWORD *)(v23[29] + 16LL));
      v26 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v25 + 16LL))(v25, v44, 0);
    }
    else
    {
      v26 = ((__int64 (__fastcall *)(_QWORD *, _QWORD, _QWORD))v24)(v23, *v21, 0);
    }
    if ( !v26 )
      goto LABEL_36;
    v29 = *v21;
LABEL_41:
    sub_3242120((_BYTE *)a1, v29, v21[2]);
    goto LABEL_36;
  }
  if ( !sub_32420F0(a1) && ((*(_BYTE *)(a1 + 101) >> 1) & 0xFu) <= 3 )
  {
    v32 = *a3;
    v33 = a3[1];
    v59.m128i_i64[0] = (__int64)v32;
    v45 = v33;
    if ( v32 != v33 )
    {
      while ( *v32 != 159 )
      {
        v32 += (unsigned int)sub_AF4160((unsigned __int64 **)&v59);
        v59.m128i_i64[0] = (__int64)v32;
        if ( v45 == v32 )
          goto LABEL_19;
      }
      goto LABEL_8;
    }
  }
LABEL_19:
  if ( *(_DWORD *)(a1 + 32) > 1u )
  {
LABEL_8:
    *(_BYTE *)(a1 + 100) &= 0xF8u;
    *(_DWORD *)(a1 + 32) = 0;
    return 0;
  }
  v52 = **(_QWORD **)(a1 + 24);
  v13 = (*(__int64 (__fastcall **)(__int64, _QWORD *, _QWORD))(*(_QWORD *)a1 + 80LL))(a1, a2, a4);
  v14 = 0;
  v15 = v13;
  if ( v47 )
  {
    v16 = *v9;
    v17 = 0;
    if ( *v9 == 35 )
    {
      if ( v9[1] > 0x7FFFFFFF )
        goto LABEL_24;
      v30 = *a3;
      v14 = v9[1];
      if ( *a3 == a3[1] )
        goto LABEL_24;
      v49 = v9[1];
      v31 = sub_AF4160(a3);
      v17 = v49;
      *a3 = &v30[v31];
      v16 = *v9;
    }
    if ( v16 != 16 )
      goto LABEL_23;
    v34 = *a3;
    if ( *a3 == a3[1] )
      goto LABEL_23;
    v50 = v17;
    v35 = v9[1];
    v59.m128i_i64[0] = (__int64)*a3;
    v36 = sub_AF4160((unsigned __int64 **)&v59);
    v17 = v50;
    v37 = &v34[v36];
    if ( a3[1] == v37 )
      goto LABEL_23;
    v38 = *v37;
    if ( v35 <= 0x7FFFFFFF && v38 == 34 )
    {
      v39 = v35;
    }
    else
    {
      if ( v38 != 28 || *(_WORD *)(a1 + 96) || v35 > 0x80000000 )
        goto LABEL_23;
      v39 = -(int)v35;
    }
    v51 = v39;
    v40 = *a3;
    v41 = &v40[(unsigned int)sub_AF4160(a3)];
    *a3 = v41;
    v42 = sub_AF4160(a3);
    v17 = v51;
    *a3 = &v41[v42];
LABEL_23:
    v14 = v17;
  }
LABEL_24:
  if ( v15 )
    sub_3242270(a1, v14);
  else
    sub_32421D0(a1, v52, v14);
  *(_DWORD *)(a1 + 32) = 0;
  v18 = *a3;
  if ( *a3 != a3[1] )
  {
    v56.m128i_i64[0] = (__int64)*a3;
    v19 = *(_WORD *)(a1 + 96) == 0;
    v56.m128i_i8[8] = 1;
    v59 = _mm_loadu_si128(&v56);
    if ( !v19 )
    {
LABEL_47:
      if ( *v18 != 4096 )
        sub_3243730(a1);
    }
  }
  return v46;
}
