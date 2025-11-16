// Function: sub_68C540
// Address: 0x68c540
//
__int64 __fastcall sub_68C540(const __m128i *a1, _QWORD *a2, _DWORD *a3, __int64 a4, _QWORD *a5)
{
  __int64 v8; // r14
  int v9; // eax
  _QWORD *v10; // rax
  _QWORD *v11; // rdi
  _DWORD *v12; // r12
  __int64 v13; // r12
  __int64 v14; // r12
  _QWORD *i; // r15
  __int64 v16; // r15
  char v17; // dl
  __int64 v18; // rax
  char v19; // dl
  _QWORD *v20; // rax
  int v21; // eax
  __int64 v22; // rax
  __int64 v23; // rax
  int v24; // edi
  __int8 v25; // al
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 j; // rdx
  __int64 v29; // rax
  __int64 v31; // rax
  __int64 v32; // rdi
  unsigned __int16 v33; // ax
  __int64 v34; // rax
  __int64 v35; // [rsp-8h] [rbp-1A8h]
  _OWORD v37[5]; // [rsp+50h] [rbp-150h] BYREF
  __m128i v38; // [rsp+A0h] [rbp-100h]
  __m128i v39; // [rsp+B0h] [rbp-F0h]
  __m128i v40; // [rsp+C0h] [rbp-E0h]
  __m128i v41; // [rsp+D0h] [rbp-D0h]
  __m128i v42; // [rsp+E0h] [rbp-C0h]
  __m128i v43; // [rsp+F0h] [rbp-B0h]
  __m128i v44; // [rsp+100h] [rbp-A0h]
  __m128i v45; // [rsp+110h] [rbp-90h]
  __m128i v46; // [rsp+120h] [rbp-80h]
  __m128i v47; // [rsp+130h] [rbp-70h]
  __m128i v48; // [rsp+140h] [rbp-60h]
  __m128i v49; // [rsp+150h] [rbp-50h]
  __m128i v50; // [rsp+160h] [rbp-40h]

  v8 = sub_6EB5C0();
  *a5 = 0;
  if ( a2 )
  {
    v9 = *(_DWORD *)(a4 + 32);
    if ( v9 == 1 )
    {
      v11 = (_QWORD *)*a2;
      if ( *a2 )
        goto LABEL_6;
    }
    else if ( v9 == 2 )
    {
      v10 = (_QWORD *)*a2;
      if ( *a2 )
      {
        v11 = (_QWORD *)*v10;
        if ( *v10 )
        {
LABEL_6:
          v12 = (_DWORD *)sub_6E1A20(v11);
          if ( (unsigned int)sub_6E5430() )
            sub_6851C0(0x8Cu, v12);
          return v8;
        }
      }
    }
    sub_6E65B0(a2);
    v13 = a2[3];
    if ( *(_BYTE *)(v13 + 25) == 1 )
    {
      sub_6FA3A0(v13 + 8);
      v14 = *(_QWORD *)(v13 + 8);
      if ( *(_BYTE *)(v14 + 140) != 12 )
        goto LABEL_12;
    }
    else
    {
      v14 = *(_QWORD *)(v13 + 8);
      if ( *(_BYTE *)(v14 + 140) != 12 )
        goto LABEL_12;
    }
    do
      v14 = *(_QWORD *)(v14 + 160);
    while ( *(_BYTE *)(v14 + 140) == 12 );
LABEL_12:
    i = (_QWORD *)*a2;
    if ( *a2 )
    {
      sub_6E65B0(*a2);
      v16 = *(_QWORD *)(*a2 + 24LL);
      if ( *(_BYTE *)(v16 + 25) == 1 )
        sub_6FA3A0(v16 + 8);
      for ( i = *(_QWORD **)(v16 + 8); *((_BYTE *)i + 140) == 12; i = (_QWORD *)i[20] )
        ;
      v17 = *(_BYTE *)(v14 + 140);
      if ( v17 == 12 )
      {
        v18 = v14;
        do
        {
          v18 = *(_QWORD *)(v18 + 160);
          v17 = *(_BYTE *)(v18 + 140);
        }
        while ( v17 == 12 );
      }
      if ( !v17 || (v32 = v14, (unsigned int)sub_8DBE70(v14)) )
      {
LABEL_21:
        v19 = *((_BYTE *)i + 140);
        if ( v19 == 12 )
        {
          v20 = i;
          do
          {
            v20 = (_QWORD *)v20[20];
            v19 = *((_BYTE *)v20 + 140);
          }
          while ( v19 == 12 );
        }
        if ( v19 && !(unsigned int)sub_8DBE70(i) && (!(unsigned int)sub_8D2780(i) || *((_BYTE *)i + 160) != 5) )
        {
          v31 = sub_6E1A20(*a2);
          sub_6E5E80(3257, v31, i);
          return v8;
        }
        goto LABEL_25;
      }
    }
    else
    {
      if ( !*(_BYTE *)(v14 + 140) )
        goto LABEL_25;
      v32 = v14;
      if ( (unsigned int)sub_8DBE70(v14) )
        goto LABEL_25;
    }
    v33 = *(_WORD *)(v8 + 176);
    if ( v33 == 4738 )
    {
      if ( (unsigned int)sub_8D2780(v14) && (unsigned int)sub_8D27E0(v14) )
        goto LABEL_57;
    }
    else
    {
      if ( v33 > 0x1282u )
      {
        if ( v33 != 15594 && v33 != 15599 )
LABEL_52:
          sub_721090(v32);
      }
      else if ( v33 != 4452 && v33 != 4587 )
      {
        goto LABEL_52;
      }
      if ( (unsigned int)sub_8D2780(v14) && !(unsigned int)sub_8D27E0(v14) )
      {
LABEL_57:
        if ( i )
          goto LABEL_21;
LABEL_25:
        v21 = sub_72BA30(5);
        v22 = sub_732700(v21, v14, (_DWORD)i, 0, 0, 0, 0, 0);
        v23 = sub_68A000(v8, v22);
        v8 = *(_QWORD *)(v23 + 88);
        v24 = v23;
        v25 = a1[1].m128i_i8[0];
        v37[0] = _mm_loadu_si128(a1 + 4);
        v37[1] = _mm_loadu_si128(a1 + 5);
        v37[2] = _mm_loadu_si128(a1 + 6);
        v37[3] = _mm_loadu_si128(a1 + 7);
        v37[4] = _mm_loadu_si128(a1 + 8);
        if ( v25 == 2 )
        {
          v38 = _mm_loadu_si128(a1 + 9);
          v39 = _mm_loadu_si128(a1 + 10);
          v40 = _mm_loadu_si128(a1 + 11);
          v41 = _mm_loadu_si128(a1 + 12);
          v42 = _mm_loadu_si128(a1 + 13);
          v43 = _mm_loadu_si128(a1 + 14);
          v44 = _mm_loadu_si128(a1 + 15);
          v45 = _mm_loadu_si128(a1 + 16);
          v46 = _mm_loadu_si128(a1 + 17);
          v47 = _mm_loadu_si128(a1 + 18);
          v48 = _mm_loadu_si128(a1 + 19);
          v49 = _mm_loadu_si128(a1 + 20);
          v50 = _mm_loadu_si128(a1 + 21);
        }
        else if ( v25 == 5 || v25 == 1 )
        {
          v38.m128i_i64[0] = a1[9].m128i_i64[0];
        }
        sub_6EAB60(
          v24,
          (a1[1].m128i_i8[2] & 0x40) != 0,
          0,
          (unsigned int)v37 + 4,
          (unsigned int)v37 + 12,
          a1[5].m128i_i64[1],
          (__int64)a1);
        j = v35;
        if ( a1[1].m128i_i8[0] )
        {
          v29 = a1->m128i_i64[0];
          for ( j = *(unsigned __int8 *)(a1->m128i_i64[0] + 140); (_BYTE)j == 12; j = *(unsigned __int8 *)(v29 + 140) )
            v29 = *(_QWORD *)(v29 + 160);
          if ( (_BYTE)j )
            sub_6F5FA0(a1, 0, 0, 1, v26, v27);
        }
        *a5 = sub_6F6D20(a2, 0, j);
        return v8;
      }
    }
    v34 = sub_6E1A20(a2);
    sub_6E5E80(3257, v34, v14);
    return v8;
  }
  if ( (unsigned int)sub_6E5430() )
    sub_6851C0(0xA5u, a3);
  return v8;
}
