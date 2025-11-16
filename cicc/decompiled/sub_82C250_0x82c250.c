// Function: sub_82C250
// Address: 0x82c250
//
__int64 __fastcall sub_82C250(
        const __m128i **a1,
        __m128i **a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        _QWORD *a6,
        __m128i **a7,
        _DWORD *a8)
{
  const __m128i **v8; // r15
  __int64 v9; // r12
  const __m128i *v10; // rbx
  __m128i *v11; // r13
  __m128i *i; // rax
  const __m128i *v13; // rax
  __m128i *v14; // rdi
  int v15; // eax
  __m128i *v16; // rdi
  __int64 result; // rax
  int v18; // ebx
  const __m128i *v19; // rax
  __int64 v20; // rdi
  char v21; // al
  __int64 v22; // r8
  const __m128i *j; // rdi
  char v24; // al
  __int64 v25; // rdi
  char v26; // al
  char v27; // dl
  __int64 k; // rbx
  __int64 v29; // r15
  __int64 v30; // r8
  __int64 v31; // rax
  _QWORD *v32; // rdi
  __int64 v33; // rsi
  __int64 v34; // [rsp-10h] [rbp-C0h]
  __m128i *v35; // [rsp+0h] [rbp-B0h]
  _QWORD *v36; // [rsp+0h] [rbp-B0h]
  const __m128i **v37; // [rsp+8h] [rbp-A8h]
  const __m128i *v38; // [rsp+10h] [rbp-A0h]
  int v40; // [rsp+20h] [rbp-90h]
  _BOOL4 v41; // [rsp+24h] [rbp-8Ch]
  const __m128i *v45; // [rsp+40h] [rbp-70h] BYREF
  __m128i *v46; // [rsp+48h] [rbp-68h] BYREF
  __int64 v47; // [rsp+50h] [rbp-60h] BYREF
  const __m128i *v48; // [rsp+58h] [rbp-58h] BYREF
  _BYTE v49[80]; // [rsp+60h] [rbp-50h] BYREF

  v8 = a1;
  v9 = a3;
  v10 = *a1;
  v11 = *a2;
  v45 = *a1;
  v46 = v11;
  if ( a8 )
    *a8 = 0;
  v41 = 0;
  if ( !a3 || *(_BYTE *)(a3 + 16) != 3 )
    goto LABEL_5;
  v20 = *(_QWORD *)(a3 + 136);
  v41 = *(_BYTE *)(a3 + 17) == 3;
  v21 = *(_BYTE *)(v20 + 80);
  v22 = v20;
  if ( v21 == 16 )
  {
    v22 = **(_QWORD **)(v20 + 88);
    v21 = *(_BYTE *)(v22 + 80);
  }
  if ( v21 == 24 )
  {
    v22 = *(_QWORD *)(v22 + 88);
    v21 = *(_BYTE *)(v22 + 80);
  }
  if ( v21 == 20 )
  {
    if ( (*(_BYTE *)(a3 + 19) & 8) == 0 )
      goto LABEL_36;
    v32 = (_QWORD *)sub_8BFF80(v20, *(_QWORD *)(a3 + 104), v49);
    if ( !v32 )
      goto LABEL_64;
    v33 = v9;
    v9 = 0;
    v46 = (__m128i *)sub_828C40(v32, v33);
    v11 = v46;
LABEL_5:
    if ( (unsigned int)sub_8D2310(v11) )
    {
      for ( i = v46; i[8].m128i_i8[12] == 12; i = (__m128i *)i[10].m128i_i64[0] )
        ;
      if ( *(_QWORD *)(i[10].m128i_i64[1] + 40) )
      {
        v9 = 0;
        v46 = (__m128i *)sub_73F0A0(v46, *(_QWORD *)(v46[10].m128i_i64[1] + 40));
      }
    }
    if ( !(unsigned int)sub_8D32E0(v45) )
    {
      v13 = (const __m128i *)sub_6EEB30((__int64)v46, v9);
      v46 = sub_73D4C0(v13, dword_4F077C4 == 2);
      v14 = v46;
      v45 = (const __m128i *)sub_8D2220(v45);
      if ( dword_4F077C4 == 2 )
      {
        if ( (unsigned int)sub_8D23B0(v46) )
          sub_8AE000(v46);
        v14 = v46;
      }
      v15 = sub_8D23B0(v14);
      v16 = v46;
      if ( v15 )
      {
        result = sub_8D2690(v46);
        v16 = v46;
        if ( !(_DWORD)result )
        {
          v10 = v45;
          v11 = v46;
          goto LABEL_19;
        }
      }
      goto LABEL_13;
    }
    v18 = sub_8D3110(v45);
    v19 = (const __m128i *)sub_8D46C0(v45);
    v45 = v19;
    if ( !v18 || !(unsigned int)sub_8D3D40(v19) )
      goto LABEL_21;
    j = v45;
    if ( (v45[8].m128i_i8[12] & 0xFB) == 8 )
    {
      if ( (unsigned int)sub_8D4C10(v45, dword_4F077C4 != 2) )
        goto LABEL_21;
      for ( j = v45; j[8].m128i_i8[12] == 12; j = (const __m128i *)j[10].m128i_i64[0] )
        ;
    }
    if ( (j[10].m128i_i8[1] & 0x10) == 0 )
    {
      if ( v9 )
      {
        v24 = *(_BYTE *)(v9 + 17);
        if ( v24 == 1 )
        {
          if ( !sub_6ED0A0(v9) )
            goto LABEL_45;
          v24 = *(_BYTE *)(v9 + 17);
        }
        if ( v24 == 3 )
          goto LABEL_45;
      }
      if ( v41 )
      {
LABEL_45:
        v46 = (__m128i *)sub_72D600(v46);
        v16 = v46;
LABEL_13:
        if ( a6 )
          *a6 = v45;
        if ( a7 )
          *a7 = v16;
        if ( (unsigned int)sub_8D2E30(v16) )
        {
          if ( (unsigned int)sub_8D2E30(v45) )
          {
            v46 = (__m128i *)sub_8D46C0(v46);
            v45 = (const __m128i *)sub_8D46C0(v45);
            if ( !(unsigned int)sub_8D2310(v46) )
              sub_828750(&v46, &v45);
          }
        }
        v10 = v45;
        v11 = v46;
        result = 1;
        goto LABEL_19;
      }
    }
LABEL_21:
    if ( dword_4F077BC
      && !(_DWORD)qword_4F077B4
      && qword_4F077A8 <= 0x9FC3u
      && v9
      && *(_BYTE *)(v9 + 16) == 1
      && (v31 = *(_QWORD *)(v9 + 144), *(_BYTE *)(v31 + 24) == 3)
      && (*(_BYTE *)(*(_QWORD *)(v31 + 56) + 172LL) & 1) != 0
      && !v18
      && (unsigned int)sub_8D3D40(v45)
      && ((v45[8].m128i_i8[12] & 0xFB) != 8 || !(unsigned int)sub_8D4C10(v45, dword_4F077C4 != 2)) )
    {
      v46 = sub_73C570(v46, 1);
      v16 = v46;
    }
    else
    {
      sub_828750(&v46, &v45);
      v16 = v46;
    }
    goto LABEL_13;
  }
  v25 = sub_82C1B0(v22, 0, 0, (__int64)v49);
  if ( !v25 )
    goto LABEL_64;
  v38 = v10;
  v35 = 0;
  v40 = 0;
  v37 = v8;
  do
  {
    v26 = *(_BYTE *)(v25 + 80);
    if ( v26 == 16 )
    {
      v25 = **(_QWORD **)(v25 + 88);
      v26 = *(_BYTE *)(v25 + 80);
    }
    if ( v26 == 24 )
    {
      v25 = *(_QWORD *)(v25 + 88);
      v26 = *(_BYTE *)(v25 + 80);
    }
    v27 = *(_BYTE *)(v9 + 19) & 8;
    if ( v26 == 20 )
    {
      if ( !v27 )
      {
        if ( !dword_4F077BC || (unsigned int)sub_8DBE70(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v25 + 88) + 176LL) + 152LL)) )
        {
LABEL_63:
          v8 = v37;
          goto LABEL_64;
        }
        goto LABEL_54;
      }
      k = sub_8BFF80(v25, *(_QWORD *)(v9 + 104), &v48);
      if ( !k )
        goto LABEL_54;
    }
    else
    {
      if ( v27 )
        goto LABEL_54;
      for ( k = *(_QWORD *)(*(_QWORD *)(v25 + 88) + 152LL); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
        ;
    }
    v29 = 0;
    v47 = k;
    if ( dword_4F077BC )
      v29 = a5;
    v48 = v38;
    if ( (unsigned int)sub_82C250((unsigned int)&v48, (unsigned int)&v47, 0, a4, 0, 0, 0, 0)
      && (unsigned int)sub_8B5160(v47, v48, a4, v29, v30, v34) )
    {
      v36 = sub_828C40((_QWORD *)k, 0);
      if ( v40 )
        goto LABEL_63;
      v40 = 1;
      if ( *(_BYTE *)(v9 + 17) != 3 )
        k = (__int64)v36;
      v35 = (__m128i *)k;
    }
LABEL_54:
    v25 = sub_82C230(v49);
  }
  while ( v25 );
  v11 = v35;
  v8 = v37;
  if ( v40 )
  {
    v46 = v35;
    v9 = 0;
    goto LABEL_5;
  }
LABEL_64:
  v10 = v45;
  v11 = v46;
LABEL_36:
  result = 0;
  if ( a8 )
    *a8 = 1;
LABEL_19:
  *v8 = v10;
  *a2 = v11;
  return result;
}
