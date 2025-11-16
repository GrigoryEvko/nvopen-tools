// Function: sub_79F930
// Address: 0x79f930
//
__int64 __fastcall sub_79F930(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, unsigned __int64 a5, char *a6)
{
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // r14
  __int64 v12; // rdx
  char v13; // al
  unsigned int v14; // r12d
  int v16; // eax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 *v19; // r9
  unsigned int v20; // eax
  __int64 v21; // r11
  __int64 v22; // rax
  __int64 *v23; // r8
  unsigned int v24; // eax
  bool v25; // al
  __int64 v26; // rcx
  __int64 v27; // rdi
  char v28; // al
  const __m128i *v29; // rax
  __m128i **v30; // rdi
  __int64 v31; // rcx
  int v32; // eax
  __int64 v33; // rcx
  unsigned int v34; // eax
  __int64 v35; // rax
  __int64 v36; // r8
  __int64 v37; // rcx
  __int64 v38; // rdi
  char j; // al
  int v40; // eax
  const __m128i *v41; // rsi
  __int64 v42; // rcx
  __int64 k; // rax
  __int64 v44; // rax
  __int64 v45; // r8
  __int64 *v46; // rax
  __int64 v47; // rcx
  _QWORD *v48; // r11
  _QWORD *v49; // rsi
  unsigned int v50; // eax
  __int64 v51; // rax
  __int64 m; // rax
  __int64 *v53; // [rsp+8h] [rbp-68h]
  __int64 *v54; // [rsp+10h] [rbp-60h]
  __int64 v55; // [rsp+10h] [rbp-60h]
  __int64 *v56; // [rsp+10h] [rbp-60h]
  __int64 v57; // [rsp+18h] [rbp-58h]
  __int64 *v58; // [rsp+18h] [rbp-58h]
  __int64 v59; // [rsp+18h] [rbp-58h]
  __int64 v60; // [rsp+18h] [rbp-58h]
  __int64 *v61; // [rsp+18h] [rbp-58h]
  __int64 v62; // [rsp+18h] [rbp-58h]
  __int64 *v63; // [rsp+18h] [rbp-58h]
  __int64 *v64; // [rsp+20h] [rbp-50h]
  __int64 *i; // [rsp+20h] [rbp-50h]
  __int64 v66; // [rsp+20h] [rbp-50h]
  __int64 *v68; // [rsp+28h] [rbp-48h]
  const __m128i *v69[7]; // [rsp+38h] [rbp-38h] BYREF

  v9 = *(_QWORD *)(a2 + 240);
  v10 = *a4;
  v69[0] = 0;
  v11 = *(_QWORD *)(v9 + 32);
  if ( *(_BYTE *)v10 == 48 )
  {
    v12 = *(_QWORD *)(v10 + 8);
    v13 = *(_BYTE *)(v12 + 8);
    if ( v13 != 1 )
    {
      if ( v13 == 2 )
      {
        *(_BYTE *)v10 = 59;
        *(_QWORD *)(v10 + 8) = *(_QWORD *)(v12 + 32);
      }
      else
      {
        if ( v13 )
          sub_721090();
        *(_BYTE *)v10 = 6;
        *(_QWORD *)(v10 + 8) = *(_QWORD *)(v12 + 32);
      }
LABEL_8:
      if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
      {
        v14 = 0;
        sub_6855B0(0xAA1u, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
        sub_770D30(a1);
        return v14;
      }
      return 0;
    }
    *(_BYTE *)v10 = 2;
    v41 = *(const __m128i **)(v12 + 32);
    *(_QWORD *)(v10 + 8) = v41;
LABEL_42:
    v42 = v41[8].m128i_i64[0];
    v69[0] = v41;
    if ( v11 != v42 )
    {
      v66 = v42;
      v14 = sub_8D97D0(v42, v11, 32, v42, a5);
      if ( !v14 )
      {
        if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
        {
          sub_687430(0xD2Bu, a3 + 28, v11, v66, (_QWORD *)(a1 + 96));
          sub_770D30(a1);
          return v14;
        }
        return 0;
      }
      v41 = v69[0];
    }
    return (unsigned int)sub_79CCD0(a1, (__int64)v41, a5, a6, 0);
  }
  switch ( *(_BYTE *)v10 )
  {
    case 2:
      v41 = *(const __m128i **)(v10 + 8);
      goto LABEL_42;
    case 7:
      v54 = (__int64 *)sub_726770(3);
      v64 = *(__int64 **)(v10 + 8);
      v57 = v64[15];
      v16 = sub_8D2FB0(v57);
      v19 = v54;
      if ( v16 )
      {
        if ( v11 == v57 || (v20 = sub_8D97D0(v57, v11, 0, v17, v18), v21 = v57, v19 = v54, (v14 = v20) != 0) )
        {
          *((_BYTE *)v19 + 25) |= 1u;
          v58 = v19;
          v22 = sub_8D46C0(v11);
          v19 = v58;
          *v58 = v22;
          v23 = v58;
          v58[7] = (__int64)v64;
          *(__int64 *)((char *)v58 + 28) = *(_QWORD *)(a3 + 28);
LABEL_16:
          v68 = v19;
          v24 = sub_786210(a1, (_QWORD **)v23, a5, a6);
          v19 = v68;
          v14 = v24;
          v25 = v24 != 0;
          goto LABEL_17;
        }
      }
      else
      {
        v55 = v57;
        *((_BYTE *)v19 + 25) |= 1u;
        v61 = v19;
        if ( (unsigned int)sub_8D2FB0(v11) )
        {
          v51 = sub_8D46C0(v11);
          v19 = v61;
          v48 = (_QWORD *)v55;
          v49 = (_QWORD *)v51;
          *v61 = v51;
          v23 = v61;
          v61[7] = (__int64)v64;
          *(__int64 *)((char *)v61 + 28) = *(_QWORD *)(a3 + 28);
        }
        else
        {
          *v61 = v11;
          v61[7] = (__int64)v64;
          *(__int64 *)((char *)v61 + 28) = *(_QWORD *)(a3 + 28);
          v46 = sub_6ED3D0((__int64)v61, 0, 0, (__int64)v61 + 28, v45, (__int64)v61);
          v48 = (_QWORD *)v55;
          v19 = v61;
          v49 = (_QWORD *)*v46;
          v23 = v46;
        }
        if ( v48 == v49 )
          goto LABEL_16;
        v53 = v19;
        v56 = v23;
        v62 = (__int64)v48;
        v50 = sub_8D97D0(v48, v49, 32, v47, v23);
        v21 = v62;
        v23 = v56;
        v19 = v53;
        v14 = v50;
        if ( v50 )
          goto LABEL_16;
      }
      if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
      {
        v25 = 0;
        v14 = 0;
      }
      else
      {
        v63 = v19;
        sub_687430(0xD2Bu, a3 + 28, v11, v21, (_QWORD *)(a1 + 96));
        sub_770D30(a1);
        v19 = v63;
        v25 = 0;
      }
LABEL_17:
      *((_BYTE *)v19 + 24) = 38;
      v26 = qword_4F06BB0;
      qword_4F06BB0 = v19;
      v19[10] = v26;
      v27 = *v64;
      goto LABEL_18;
    case 8:
      if ( (*(_BYTE *)(*(_QWORD *)(v10 + 8) + 144LL) & 4) == 0 )
      {
        v29 = (const __m128i *)sub_724DC0();
        v30 = *(__m128i ***)(v10 + 8);
        v69[0] = v29;
        sub_73F1E0(v30, (__int64)v29);
        v14 = sub_79F800(a1, v69[0], v11, a3, a5, a6);
        sub_724E30((__int64)v69);
        return v14;
      }
      if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
      {
        sub_6855B0(0x8Bu, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
        sub_770D30(a1);
      }
      return 0;
    case 0xB:
      v31 = *(_QWORD *)(*(_QWORD *)(v10 + 8) + 152LL);
      for ( i = *(__int64 **)(v10 + 8); *(_BYTE *)(v31 + 140) == 12; v31 = *(_QWORD *)(v31 + 160) )
        ;
      v59 = v31;
      v32 = sub_8D2FB0(v11);
      v33 = v59;
      if ( v32 || (v34 = sub_8D2E30(v11), v33 = v59, (v14 = v34) != 0) )
      {
        v60 = v33;
        v35 = sub_8D46C0(v11);
        v37 = v60;
        v38 = v35;
        for ( j = *(_BYTE *)(v35 + 140); j == 12; j = *(_BYTE *)(v38 + 140) )
          v38 = *(_QWORD *)(v38 + 160);
        if ( j == 7 )
        {
          if ( v38 == v60 || (v40 = sub_8D97D0(v38, v60, 0, v60, v36), v37 = v60, v40) )
          {
            for ( k = v37; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
              ;
            if ( !*(_QWORD *)(*(_QWORD *)(k + 168) + 40LL) )
            {
              v14 = 1;
              *(_OWORD *)a5 = 0;
              *(_OWORD *)(a5 + 16) = 0;
              *(_BYTE *)(a5 + 8) = 32;
              *(_QWORD *)(a5 + 16) = i;
              v44 = -(((unsigned int)(a5 - (_DWORD)a6) >> 3) + 10);
              a6[v44] |= 1 << ((a5 - (_BYTE)a6) & 7);
              v25 = 1;
              goto LABEL_39;
            }
          }
        }
        if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
        {
          sub_687430(0xD2Bu, a3 + 28, v11, v37, (_QWORD *)(a1 + 96));
          sub_770D30(a1);
        }
LABEL_38:
        v25 = 0;
        v14 = 0;
        goto LABEL_39;
      }
      if ( !(unsigned int)sub_8D3D10(v11) )
        goto LABEL_68;
      for ( m = v59; *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
        ;
      if ( *(_QWORD *)(*(_QWORD *)(m + 168) + 40LL) )
      {
        v69[0] = (const __m128i *)sub_724DC0();
        sub_73F170(i, (__int64)v69[0]->m128i_i64);
        v14 = sub_79F800(a1, v69[0], v11, a3, a5, a6);
        sub_724E30((__int64)v69);
        v25 = v14 != 0;
      }
      else
      {
LABEL_68:
        if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
          goto LABEL_38;
        sub_687430(0xD2Bu, a3 + 28, v11, v59, (_QWORD *)(a1 + 96));
        sub_770D30(a1);
        v25 = 0;
      }
LABEL_39:
      v27 = *i;
LABEL_18:
      if ( v27 && v25 )
      {
        if ( (v28 = *(_BYTE *)(v27 + 80), v28 != 7) && v28 != 9 && (unsigned __int8)(v28 - 10) > 1u
          || *(_QWORD *)(v27 + 96) )
        {
          sub_8AD0D0(v27, 1, 0);
        }
      }
      return v14;
    case 0xD:
      return (unsigned int)sub_786210(a1, *(_QWORD ***)(v10 + 8), a5, a6);
    default:
      goto LABEL_8;
  }
}
