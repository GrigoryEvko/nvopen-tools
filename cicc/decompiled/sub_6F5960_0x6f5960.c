// Function: sub_6F5960
// Address: 0x6f5960
//
__int64 __fastcall sub_6F5960(__m128i *a1, __int64 a2, _DWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int8 v7; // al
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 i; // rbx
  __int64 v13; // rdx
  bool v14; // cl
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // ebx
  __int16 v18; // r15
  _QWORD *v19; // rax
  __int64 v20; // rdx
  __int64 v22; // r8
  __int64 v23; // r9
  __int16 v24; // bx
  int v25; // r15d
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 *v33; // rax
  __int64 v34; // [rsp+8h] [rbp-1A8h]
  int v35; // [rsp+8h] [rbp-1A8h]
  __int64 v36; // [rsp+8h] [rbp-1A8h]
  __int64 v37; // [rsp+8h] [rbp-1A8h]
  __int64 *v38; // [rsp+18h] [rbp-198h] BYREF
  _OWORD v39[4]; // [rsp+20h] [rbp-190h] BYREF
  __m128i v40; // [rsp+60h] [rbp-150h]
  __m128i v41; // [rsp+70h] [rbp-140h]
  __m128i v42; // [rsp+80h] [rbp-130h]
  __m128i v43; // [rsp+90h] [rbp-120h]
  __m128i v44; // [rsp+A0h] [rbp-110h]
  __m128i v45; // [rsp+B0h] [rbp-100h]
  __m128i v46; // [rsp+C0h] [rbp-F0h]
  __m128i v47; // [rsp+D0h] [rbp-E0h]
  __m128i v48; // [rsp+E0h] [rbp-D0h]
  __m128i v49; // [rsp+F0h] [rbp-C0h]
  __m128i v50; // [rsp+100h] [rbp-B0h]
  __m128i v51; // [rsp+110h] [rbp-A0h]
  __m128i v52; // [rsp+120h] [rbp-90h]
  __m128i v53; // [rsp+130h] [rbp-80h]
  __m128i v54; // [rsp+140h] [rbp-70h]
  __m128i v55; // [rsp+150h] [rbp-60h]
  __m128i v56; // [rsp+160h] [rbp-50h]
  __m128i v57; // [rsp+170h] [rbp-40h]

  v38 = (__int64 *)sub_724DC0(a1, a2, a3, a4, a5, a6);
  v39[0] = _mm_loadu_si128(a1);
  v39[1] = _mm_loadu_si128(a1 + 1);
  v7 = a1[1].m128i_i8[0];
  v39[2] = _mm_loadu_si128(a1 + 2);
  v39[3] = _mm_loadu_si128(a1 + 3);
  v40 = _mm_loadu_si128(a1 + 4);
  v41 = _mm_loadu_si128(a1 + 5);
  v42 = _mm_loadu_si128(a1 + 6);
  v43 = _mm_loadu_si128(a1 + 7);
  v44 = _mm_loadu_si128(a1 + 8);
  if ( v7 == 2 )
  {
    v45 = _mm_loadu_si128(a1 + 9);
    v46 = _mm_loadu_si128(a1 + 10);
    v47 = _mm_loadu_si128(a1 + 11);
    v48 = _mm_loadu_si128(a1 + 12);
    v49 = _mm_loadu_si128(a1 + 13);
    v50 = _mm_loadu_si128(a1 + 14);
    v51 = _mm_loadu_si128(a1 + 15);
    v52 = _mm_loadu_si128(a1 + 16);
    v53 = _mm_loadu_si128(a1 + 17);
    v54 = _mm_loadu_si128(a1 + 18);
    v55 = _mm_loadu_si128(a1 + 19);
    v56 = _mm_loadu_si128(a1 + 20);
    v57 = _mm_loadu_si128(a1 + 21);
  }
  else if ( v7 == 5 || v7 == 1 )
  {
    v45.m128i_i64[0] = a1[9].m128i_i64[0];
  }
  v8 = sub_6F6F40(a1, 0);
  v9 = sub_72B0F0(v8, 0);
  if ( !v9 )
  {
    v15 = qword_4D03C50;
    if ( (*(_BYTE *)(qword_4D03C50 + 18LL) & 8) != 0 && !(_DWORD)a2 )
      goto LABEL_21;
    if ( dword_4F04C44 == -1 )
    {
      v14 = (_DWORD)a2 == 0;
LABEL_26:
      v15 = qword_4D03C50;
      goto LABEL_31;
    }
    v14 = (_DWORD)a2 == 0;
LABEL_31:
    if ( (*(_BYTE *)(v15 + 19) & 0x40) == 0 || !v14 )
    {
      v17 = 0;
      if ( !a3 )
      {
        v33 = sub_6ED3D0(v8, 0, 0, (__int64)a1[4].m128i_i64 + 4, v10, v11);
        sub_6E70E0(v33, (__int64)a1);
        goto LABEL_40;
      }
LABEL_34:
      v18 = *(_WORD *)(v8 + 48);
      v35 = *(_DWORD *)(v8 + 44);
      if ( !(unsigned int)sub_8D3D40(*(_QWORD *)v8) )
      {
        v31 = sub_72D2E0(*(_QWORD *)v8, 0);
        v19 = (_QWORD *)sub_73DBF0(0, v31, v8);
        *(_QWORD *)((char *)v19 + 28) = *(_QWORD *)a3;
        v32 = *(_QWORD *)a3;
        *((_DWORD *)v19 + 11) = v35;
        *(_QWORD *)((char *)v19 + 36) = v32;
        *((_WORD *)v19 + 24) = v18;
        if ( !v17 )
        {
          sub_6E70E0(v19, (__int64)a1);
          goto LABEL_39;
        }
LABEL_36:
        if ( *((_BYTE *)v19 + 24) == 2 )
        {
LABEL_39:
          *(__int64 *)((char *)v40.m128i_i64 + 4) = *(_QWORD *)a3;
          goto LABEL_40;
        }
        goto LABEL_37;
      }
      v19 = (_QWORD *)sub_73DBF0(0, *(_QWORD *)&dword_4D03B80, v8);
      *(_QWORD *)((char *)v19 + 28) = *(_QWORD *)a3;
      v20 = *(_QWORD *)a3;
      *((_DWORD *)v19 + 11) = v35;
      *(_QWORD *)((char *)v19 + 36) = v20;
      *((_WORD *)v19 + 24) = v18;
      if ( v17 )
        goto LABEL_36;
LABEL_57:
      sub_6E70E0(v19, (__int64)a1);
      if ( v17 )
        sub_6F4B70(a1, (__int64)a1, v27, v28, v29, v30);
      goto LABEL_39;
    }
    sub_6E68E0(0x1Cu, (__int64)a1);
    goto LABEL_38;
  }
  for ( i = *(_QWORD *)(v9 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( (_DWORD)a2 )
    goto LABEL_25;
  if ( (*(_BYTE *)(*(_QWORD *)(i + 168) + 20LL) & 2) != 0 )
  {
    v37 = v9;
    sub_6E5470(*(_QWORD *)(i + 104), &a1[4].m128i_i32[1]);
    v9 = v37;
  }
  if ( (*(_BYTE *)(v9 + 193) & 4) != 0 )
  {
    v36 = v9;
    sub_6DEC10(v9);
    v9 = v36;
  }
  if ( (a1[1].m128i_i8[2] & 8) != 0 && ((a1[1].m128i_i8[2] & 0x40) == 0 || !a3) )
  {
    v13 = **(_QWORD **)(i + 168);
    if ( v13 )
    {
      if ( (*(_BYTE *)(v13 + 35) & 1) != 0 )
      {
        v34 = v9;
        if ( a3 )
          sub_6851C0(0xCA5u, a3);
        else
          sub_6851C0(0xCA6u, &a1[4].m128i_i32[1]);
        v9 = v34;
      }
    }
  }
  if ( (*(_BYTE *)(qword_4D03C50 + 18LL) & 8) == 0 )
  {
LABEL_25:
    if ( dword_4F04C44 == -1 )
    {
      v16 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      v14 = (_DWORD)a2 == 0;
      if ( (*(_BYTE *)(v16 + 6) & 6) == 0 && *(_BYTE *)(v16 + 4) != 12 )
        goto LABEL_26;
    }
    while ( *(_BYTE *)(i + 140) == 12 )
      i = *(_QWORD *)(i + 160);
    v14 = (_DWORD)a2 == 0;
    if ( !*(_QWORD *)(*(_QWORD *)(i + 168) + 40LL)
      && (*(_BYTE *)(v9 + 89) & 4) != 0
      && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v9 + 40) + 32LL) + 177LL) & 0x20) != 0 )
    {
      goto LABEL_21;
    }
    v15 = qword_4D03C50;
    goto LABEL_31;
  }
LABEL_21:
  if ( !(unsigned int)sub_717510(v8, v38, (_DWORD)a2 == 0) )
  {
    v14 = (_DWORD)a2 == 0;
    v15 = qword_4D03C50;
    goto LABEL_31;
  }
  if ( unk_4F07734 && *((_BYTE *)v38 + 173) == 12 && a3 )
  {
    sub_6E83C0(v38, 0);
    v24 = *(_WORD *)(v8 + 48);
    v25 = *(_DWORD *)(v8 + 44);
    v19 = (_QWORD *)sub_73DBF0(0, *(_QWORD *)&dword_4D03B80, v8);
    *(_QWORD *)((char *)v19 + 28) = *(_QWORD *)a3;
    v26 = *(_QWORD *)a3;
    *((_DWORD *)v19 + 11) = v25;
    *(_QWORD *)((char *)v19 + 36) = v26;
    *((_WORD *)v19 + 24) = v24;
    v17 = 1;
    goto LABEL_57;
  }
  sub_6E6A50((__int64)v38, (__int64)a1);
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) && *((_BYTE *)v38 + 173) != 12 )
  {
    if ( !a3 )
    {
      v19 = sub_6ED3D0(v8, 0, 0, (__int64)a1[4].m128i_i64 + 4, v22, v23);
      if ( *((_BYTE *)v19 + 24) == 2 )
        goto LABEL_40;
LABEL_37:
      a1[18].m128i_i64[0] = (__int64)v19;
      goto LABEL_38;
    }
    v17 = 1;
    goto LABEL_34;
  }
LABEL_38:
  if ( a3 )
    goto LABEL_39;
LABEL_40:
  sub_6E4EE0((__int64)a1, (__int64)v39);
  return sub_724E30(&v38);
}
