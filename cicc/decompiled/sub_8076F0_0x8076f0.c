// Function: sub_8076F0
// Address: 0x8076f0
//
__int64 __fastcall sub_8076F0(__int64 a1)
{
  __int64 i; // r12
  __int64 *v3; // r14
  __int64 v4; // rdi
  __int64 v5; // r12
  __int64 v6; // rcx
  __int64 *v7; // r15
  __m128i *v8; // r14
  __int64 v9; // rbx
  __m128i *v10; // rax
  __m128i *v11; // rax
  const __m128i *v12; // rdi
  int v13; // esi
  __m128i *v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r15
  __int64 v17; // rbx
  __m128i *v18; // rax
  __m128i *v19; // rdi
  __int64 *v20; // rax
  __int64 result; // rax
  _QWORD *v22; // rax
  __int64 *v23; // r15
  _QWORD *v24; // rax
  __int64 v25; // rsi
  _QWORD *v26; // rax
  _BYTE *v27; // rax
  __int64 v28; // rax
  _BYTE *v29; // rdi
  __int64 v30; // r15
  _QWORD *v31; // rax
  __int64 v32; // rsi
  _BYTE *v33; // rax
  _BYTE *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rsi
  _QWORD *v40; // rax
  _BYTE *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  _QWORD *v46; // rax
  __int64 *v47; // rax
  __int64 v48; // rsi
  _QWORD *v49; // rax
  _BYTE *v50; // rax
  __int64 *v51; // rax
  __int64 v52; // rax
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // rax
  _DWORD *v56; // rax
  __m128i *v57; // [rsp+8h] [rbp-148h]
  __int64 v58; // [rsp+18h] [rbp-138h]
  __int64 v59; // [rsp+18h] [rbp-138h]
  __int64 v60; // [rsp+18h] [rbp-138h]
  __int64 v61; // [rsp+20h] [rbp-130h]
  _QWORD *v62; // [rsp+20h] [rbp-130h]
  __int64 *v63; // [rsp+20h] [rbp-130h]
  _QWORD *v64; // [rsp+20h] [rbp-130h]
  __int64 *v65; // [rsp+28h] [rbp-128h]
  __m128i *v66; // [rsp+28h] [rbp-128h]
  unsigned int v67; // [rsp+34h] [rbp-11Ch] BYREF
  __m128i *v68; // [rsp+38h] [rbp-118h] BYREF
  int v69[8]; // [rsp+40h] [rbp-110h] BYREF
  _BYTE v70[240]; // [rsp+60h] [rbp-F0h] BYREF

  for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v3 = sub_7F54F0(a1, 1, 0, &v67);
  sub_7F6C60((__int64)v3, v67, (__int64)v70);
  v4 = i;
  v5 = 0;
  v6 = sub_7E6700(v4);
  v7 = **(__int64 ***)(*(_QWORD *)(a1 + 152) + 168LL);
  if ( v7 )
  {
    v65 = v3;
    v8 = 0;
    v9 = v6;
    while ( 1 )
    {
      v12 = (const __m128i *)v7[1];
      v13 = (*((_DWORD *)v7 + 8) >> 11) & 0x7F;
      if ( (__int64 *)v9 == v7 )
      {
        if ( *(_QWORD *)(a1 + 288) || *(_QWORD *)(a1 + 296) )
          v13 = (*((_DWORD *)v7 + 8) >> 11) & 0x7E;
        v14 = sub_73C570(v12, v13);
        v11 = sub_7E2270((__int64)v14);
        v11[10].m128i_i8[12] |= 1u;
        v5 = (__int64)v11;
        if ( v8 )
        {
LABEL_6:
          v8[7].m128i_i64[0] = (__int64)v11;
          v11[7].m128i_i64[0] = 0;
          v7 = (__int64 *)*v7;
          if ( !v7 )
            goto LABEL_14;
          goto LABEL_7;
        }
      }
      else
      {
        v10 = sub_73C570(v12, v13);
        v11 = sub_7E2270((__int64)v10);
        if ( v8 )
          goto LABEL_6;
      }
      v65[5] = (__int64)v11;
      v11[7].m128i_i64[0] = 0;
      v7 = (__int64 *)*v7;
      if ( !v7 )
      {
LABEL_14:
        v3 = v65;
        break;
      }
LABEL_7:
      v8 = v11;
    }
  }
  v15 = *(_QWORD *)(a1 + 280);
  v61 = *(_QWORD *)(a1 + 272);
  v16 = sub_7F8700(*(_QWORD *)(v61 + 152));
  v17 = sub_7F8700(*(_QWORD *)(v15 + 152));
  v18 = (__m128i *)sub_726700(19);
  v18->m128i_i64[0] = v16;
  v66 = v18;
  if ( (unsigned int)sub_8D2600(v16) )
  {
    v68 = v66;
    v57 = 0;
  }
  else
  {
    v57 = sub_7E7CA0(v66->m128i_i64[0]);
    v68 = (__m128i *)sub_73E830((__int64)v57);
  }
  if ( !(unsigned int)sub_8D32B0(v16)
    || (v52 = sub_8D46C0(v16), !(unsigned int)sub_8D3A70(v52))
    || v16 == v17
    || (unsigned int)sub_8D97D0(v16, v17, 0, v53, v54) )
  {
    v19 = v68;
  }
  else
  {
    v59 = sub_8D46C0(v17);
    v55 = sub_8D46C0(v16);
    v60 = sub_8D5B70(v55, v59, 0);
    if ( v60 )
    {
      v56 = (_DWORD *)sub_8D46C0(v17);
      sub_6E7420(v60, v56, 0, 0, 0, 1, 0, (__int64 *)&v68, (_DWORD *)(v61 + 64), v69);
    }
    else
    {
      v68 = (__m128i *)sub_73E110((__int64)v68, v17);
    }
    v19 = v68;
  }
  sub_7EE560(v19, 0);
  v58 = *(_QWORD *)(v3[10] + 72);
  if ( (unsigned int)sub_8D2600(v16) )
  {
    sub_7E1740(v3[10], (__int64)v69);
    sub_7E69E0(v68, v69);
    v20 = 0;
    v68 = 0;
  }
  else
  {
    v20 = (__int64 *)v68;
  }
  *(_QWORD *)(v58 + 48) = v20;
  sub_7E1740(v3[10], (__int64)v69);
  if ( (*(_BYTE *)(v61 + 205) & 2) != 0 )
    sub_7E5120(a1);
  if ( !*(_QWORD *)(a1 + 288) )
  {
    if ( !*(_QWORD *)(a1 + 296) )
      goto LABEL_25;
    v30 = 0;
    goto LABEL_33;
  }
  v22 = sub_73E830(v5);
  v23 = (__int64 *)sub_7E23D0(v22);
  v24 = sub_73A830(*(_QWORD *)(a1 + 288), unk_4F06A60);
  v25 = *v23;
  v23[2] = (__int64)v24;
  v26 = sub_73DBF0(0x32u, v25, (__int64)v23);
  v27 = sub_73E130(v26, *(_QWORD *)(v5 + 120));
  v28 = sub_7E2BE0(v5, (__int64)v27);
  v29 = (_BYTE *)v28;
  if ( *(_QWORD *)(a1 + 296) )
  {
    v30 = v28;
LABEL_33:
    v62 = sub_73E830(v5);
    v31 = (_QWORD *)sub_7E1DC0();
    v32 = sub_72D2E0(v31);
    v33 = sub_73E130(v62, v32);
    v34 = sub_73DCD0(v33);
    v63 = sub_731370((__int64)v34, v32, v35, v36, v37, v38);
    v63[2] = (__int64)sub_73A830(*(_QWORD *)(a1 + 296), unk_4F06A60);
    v39 = *v63;
    v40 = sub_73DBF0(0x32u, *v63, (__int64)v63);
    v41 = sub_73DCD0(v40);
    v64 = sub_731370((__int64)v41, v39, v42, v43, v44, v45);
    v46 = sub_73E830(v5);
    v47 = (__int64 *)sub_7E23D0(v46);
    v48 = *v47;
    v47[2] = (__int64)v64;
    v49 = sub_73DBF0(0x32u, v48, (__int64)v47);
    v50 = sub_73E130(v49, *(_QWORD *)(v5 + 120));
    v51 = (__int64 *)sub_7E2BE0(v5, (__int64)v50);
    v29 = v51;
    if ( v30 )
      v29 = sub_73DF90(v30, v51);
  }
  sub_7E69E0(v29, v69);
LABEL_25:
  if ( v57 )
    sub_7E6AB0((__int64)v57, (__int64)v66, v69);
  sub_7FB010((__int64)v3, v67, (__int64)v70);
  result = dword_4F068EC;
  if ( dword_4F068EC && *(char *)(a1 + 192) < 0 && !*(_BYTE *)(a1 + 172) )
    return sub_7604D0(a1, 0xBu);
  return result;
}
