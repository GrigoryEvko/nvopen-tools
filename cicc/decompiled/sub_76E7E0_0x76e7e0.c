// Function: sub_76E7E0
// Address: 0x76e7e0
//
_QWORD *__fastcall sub_76E7E0(__int64 a1, _DWORD *a2)
{
  int v3; // r12d
  _QWORD *v4; // r14
  const __m128i *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  _UNKNOWN *__ptr32 *v8; // r8
  const __m128i *v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // r12
  int v14; // eax
  __int64 v15; // r13
  unsigned __int8 v16; // r12
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  const __m128i *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  _UNKNOWN *__ptr32 *v24; // r8
  __int64 v25; // r15
  const __m128i *v26; // rax
  __int64 v27; // rax
  const __m128i *v28; // r15
  const __m128i *v29; // rsi
  _BOOL8 v30; // r12
  __int64 v31; // r13
  __int64 v32; // rax
  __int8 v33; // al
  __int8 v34; // al
  __int8 v35; // dl
  int v36; // [rsp+10h] [rbp-40h] BYREF
  int v37; // [rsp+14h] [rbp-3Ch] BYREF
  const __m128i *v38; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(unsigned __int8 *)(a1 + 24);
  v4 = *(_QWORD **)a1;
  v5 = (const __m128i *)sub_724DC0();
  v38 = v5;
  if ( v3 == 3 )
  {
    v12 = *(_QWORD *)(a1 + 56);
    if ( (*(_BYTE *)(a1 + 25) & 1) != 0 )
    {
      v27 = sub_76DBC0(v12);
      if ( v27 )
        *(_QWORD *)(a1 + 56) = v27;
      else
        *a2 = 1;
    }
    else
    {
      v13 = *(_QWORD *)(v12 + 264);
      if ( v13 )
      {
        v14 = *(_DWORD *)(v13 + 16);
        if ( v14 )
        {
          if ( v14 == 1 )
          {
            *(_QWORD *)(a1 + 56) = *(_QWORD *)(v13 + 24);
            *(_BYTE *)(v13 + 56) = 1;
          }
          else
          {
            if ( v14 != 2 )
              sub_721090();
            v25 = *(_QWORD *)(v13 + 24);
            if ( *(_BYTE *)(v25 + 24) == 2 )
            {
              v31 = *(_QWORD *)(a1 + 8);
              sub_7264E0(a1, 2);
              v32 = *(_QWORD *)(v25 + 56);
              *(_QWORD *)(a1 + 8) = v31;
              *(_QWORD *)(a1 + 56) = v32;
            }
            else
            {
              v26 = (const __m128i *)sub_73F620(*(const __m128i **)(v13 + 24), a2);
              sub_730620(a1, v26);
              if ( (unsigned int)sub_8D3D10(v4) )
                *(_QWORD *)a1 = v4;
            }
          }
          *(_BYTE *)(v13 + 57) = 1;
        }
      }
    }
LABEL_4:
    if ( (*(_BYTE *)(a1 + 26) & 0x10) == 0 || !(unsigned int)sub_7E6B40(a1, 1, 1, 1, &v37) || !v37 )
      return sub_724E30((__int64)&v38);
    goto LABEL_27;
  }
  if ( v3 == 2 )
  {
    v10 = *(const __m128i **)(a1 + 56);
    if ( (v10[-1].m128i_i8[8] & 1) == 0 )
    {
      v11 = sub_73A460(v10, (__int64)a2, v6, v7, v8);
      *(_QWORD *)(a1 + 56) = v11;
      *(_QWORD *)(v11 + 144) = 0;
    }
    goto LABEL_4;
  }
  if ( v3 != 1 )
    goto LABEL_4;
  v15 = *(_QWORD *)(a1 + 72);
  v16 = *(_BYTE *)(a1 + 56);
  if ( *(_BYTE *)(v15 + 24) != 2 )
  {
LABEL_17:
    if ( (unsigned __int8)(v16 - 58) <= 1u && *(_BYTE *)(a1 + 57) == 6 )
    {
      if ( (unsigned int)sub_7E6B40(*(_QWORD *)(a1 + 72), 1, 1, 1, &v37) )
      {
        if ( v37 )
        {
          v17 = *(_QWORD *)(v15 + 16);
          if ( *(_BYTE *)(v17 + 24) == 2
            && sub_70FCE0(*(_QWORD *)(v17 + 56))
            && (unsigned int)sub_711520(*(_QWORD *)(v17 + 56), 1, v18, v19, v20) )
          {
            sub_72BAF0((__int64)v38, v16 == 59, 5u);
            if ( (*(_BYTE *)(a1 + 26) & 0x10) == 0 || !(unsigned int)sub_7E6B40(a1, 1, 1, 1, &v37) || !v37 )
              goto LABEL_28;
            goto LABEL_27;
          }
        }
      }
    }
    goto LABEL_4;
  }
  v28 = *(const __m128i **)(v15 + 16);
  if ( v28 )
  {
    if ( v28[1].m128i_i8[8] != 2 )
      goto LABEL_17;
    v36 = 1;
    v37 = 0;
    v29 = *(const __m128i **)(v15 + 56);
    v28 = (const __m128i *)v28[3].m128i_i64[1];
  }
  else
  {
    v37 = 0;
    v29 = *(const __m128i **)(v15 + 56);
    v36 = 1;
  }
  if ( v16 > 0x1Du )
  {
    if ( v16 > 0x2Bu )
    {
      if ( (unsigned __int8)(v16 - 50) > 0xDu )
        goto LABEL_42;
    }
    else if ( v16 <= 0x26u )
    {
      goto LABEL_42;
    }
    v33 = v29[10].m128i_i8[13];
    if ( v33 == 1 || v33 == 6 )
    {
      v34 = v28[10].m128i_i8[13];
      if ( v34 == 6 || v34 == 1 )
      {
        if ( (unsigned int)sub_8D3D10(v4) )
          v4 = sub_72BA30(unk_4D03F80);
        sub_713ED0(v16, v29, v28, (__int64)v4, (__int64)v38, 0, 0, &v36, &v37, 0, dword_4D03F38);
        v30 = v36 == 0;
LABEL_60:
        if ( (*(_BYTE *)(a1 + 26) & 0x10) == 0 || !(unsigned int)sub_7E6B40(a1, 1, 1, 1, &v37) )
          goto LABEL_46;
        goto LABEL_45;
      }
    }
  }
  else if ( v16 > 0x1Bu || v16 == 26 && *(_BYTE *)(a1 + 57) == 2 )
  {
    v35 = v29[10].m128i_i8[13];
    if ( v35 == 6 || v35 == 1 )
    {
      sub_712770(v16, v29, (__int64)v4, (__int64)v5, 0, 0, &v36, &v37, 0, dword_4D03F38);
      v30 = v36 == 0;
      goto LABEL_60;
    }
  }
LABEL_42:
  if ( (*(_BYTE *)(a1 + 26) & 0x10) != 0 && (unsigned int)sub_7E6B40(a1, 1, 1, 1, &v37) )
  {
    LODWORD(v30) = 0;
LABEL_45:
    if ( v37 )
    {
LABEL_27:
      sub_72BAF0((__int64)v38, 1, 5u);
LABEL_28:
      v21 = v38;
      v38[8].m128i_i64[0] = (__int64)v4;
      v21[9].m128i_i64[0] = 0;
      sub_7264E0(a1, 2);
      *(_QWORD *)(a1 + 56) = sub_73A460(v38, 2, v22, v23, v24);
      return sub_724E30((__int64)&v38);
    }
LABEL_46:
    if ( !v30 )
      return sub_724E30((__int64)&v38);
    goto LABEL_28;
  }
  return sub_724E30((__int64)&v38);
}
