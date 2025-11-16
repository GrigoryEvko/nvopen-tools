// Function: sub_7E5FC0
// Address: 0x7e5fc0
//
__int64 __fastcall sub_7E5FC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, unsigned int a6)
{
  __int64 v9; // rax
  bool v10; // zf
  int v11; // r15d
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 result; // rax
  int v15; // eax
  __int64 *v16; // r11
  __int64 *v17; // r10
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 *v20; // r10
  __int64 *v21; // r11
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // rdx
  char v25; // al
  _BYTE *v26; // rax
  __int64 v27; // rdx
  unsigned int v28; // eax
  __int64 v29; // rcx
  __int64 v30; // rdi
  int v31; // [rsp+Ch] [rbp-64h]
  int v32; // [rsp+Ch] [rbp-64h]
  __int64 *v33; // [rsp+10h] [rbp-60h]
  int v34; // [rsp+10h] [rbp-60h]
  __int64 v35; // [rsp+10h] [rbp-60h]
  __int64 *v36; // [rsp+10h] [rbp-60h]
  __int64 *v38; // [rsp+18h] [rbp-58h]
  __int64 *v39; // [rsp+18h] [rbp-58h]
  __int16 v40; // [rsp+2Ah] [rbp-46h] BYREF
  int v41; // [rsp+2Ch] [rbp-44h] BYREF
  __int64 v42; // [rsp+30h] [rbp-40h] BYREF
  __int64 v43; // [rsp+38h] [rbp-38h] BYREF

  v9 = *(_QWORD *)(a4 + 120);
  v42 = 0;
  v10 = *(_QWORD *)(v9 + 176) == 0;
  v43 = 0;
  v11 = v10;
  if ( !v10 && (*(_BYTE *)(a4 + 174) & 2) == 0 && a5 )
    v11 = 1;
  if ( a2 )
  {
    v12 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 168LL);
  }
  else
  {
    v32 = a5;
    v35 = *(_QWORD *)(a1 + 168);
    v28 = sub_7DB8B0(a4);
    v12 = v35;
    a5 = v32;
    if ( v28 != 11 )
    {
      v29 = qword_4F06C80[v28];
      if ( a1 != v29 && v29 )
        *(_WORD *)(v35 + 44) = *(_WORD *)(*(_QWORD *)(v29 + 168) + 44LL);
      result = v32 | v28;
      if ( !(_DWORD)result )
      {
        if ( !v11 )
          return result;
        if ( *(_QWORD *)(a4 + 8) )
          goto LABEL_11;
LABEL_26:
        v27 = 0;
        if ( a3 )
          v27 = a2;
        v34 = a5;
        sub_7E3260(a4, a1, v27, a3);
        a5 = v34;
        goto LABEL_11;
      }
    }
  }
  v31 = a5;
  v13 = (unsigned __int16)(*(_WORD *)(v12 + 44) + 1);
  if ( !(_WORD)v13 )
    v13 = 0;
  *(_QWORD *)(*(_QWORD *)(a4 + 120) + 176LL) += sub_7E3470(a1, a2) + v13;
  *(_QWORD *)(*(_QWORD *)(a4 + 120) + 128LL) = 0;
  result = sub_8D6090(*(_QWORD *)(a4 + 120));
  a5 = v31;
  if ( !v11 )
  {
    if ( !v31 )
      return result;
    *(_BYTE *)(a4 + 174) |= 2u;
    sub_7296C0(&v41);
    v16 = &v43;
    v17 = &v42;
    goto LABEL_16;
  }
  if ( !*(_QWORD *)(a4 + 8) )
    goto LABEL_26;
LABEL_11:
  result = a6;
  if ( a6 )
  {
    v15 = *(unsigned __int8 *)(a4 + 88);
    *(_BYTE *)(a4 + 168) &= 0xF8u;
    *(_BYTE *)(a4 + 136) = 2;
    result = v15 & 0xFFFFFF8F | 0x10;
    *(_BYTE *)(a4 + 88) = result;
    if ( !a5 )
      return result;
    goto LABEL_30;
  }
  if ( !a5 )
    return result;
  *(_BYTE *)(a4 + 88) |= 4u;
  *(_BYTE *)(a4 + 136) = 0;
  if ( !a3 )
  {
    sub_7E4C10(a4);
LABEL_30:
    v25 = *(_BYTE *)(a4 + 174);
    goto LABEL_25;
  }
  v24 = sub_7E3E50(*(_QWORD *)(a3 + 56));
  v25 = *(_BYTE *)(v24 + 174) & 1 | *(_BYTE *)(a4 + 174) & 0xFE;
  *(_BYTE *)(a4 + 174) = v25;
  *(_QWORD *)(a4 + 240) = *(_QWORD *)(v24 + 240);
LABEL_25:
  *(_BYTE *)(a4 + 174) = v25 | 2;
  sub_7296C0(&v41);
  v26 = sub_724D50(10);
  v17 = (__int64 *)(v26 + 176);
  v16 = (__int64 *)(v26 + 184);
  *((_QWORD *)v26 + 16) = *(_QWORD *)(a4 + 120);
  *(_BYTE *)(a4 + 177) = 1;
  *(_QWORD *)(a4 + 184) = v26;
LABEL_16:
  if ( a3 )
  {
    v33 = v16;
    v38 = v17;
    if ( a2 )
    {
      v18 = sub_8E5650(a2);
      sub_7E2E30(*(_QWORD *)(a3 + 104) - *(_QWORD *)(v18 + 104), 0, 1, v38, v33, 0, a1);
      v19 = sub_8E5650(a2);
      v20 = v38;
      v21 = v33;
      v22 = v19;
    }
    else
    {
      sub_7E2E30(0, 0, 1, v17, v16, 0, a1);
      v20 = v38;
      v22 = a3;
      v21 = v33;
    }
  }
  else
  {
    v30 = 0;
    if ( a2 )
      v30 = -*(_QWORD *)(a2 + 104);
    v36 = v16;
    v39 = v17;
    sub_7E2E30(v30, 0, 1, v17, v16, 0, a1);
    v20 = v39;
    v22 = a2;
    v21 = v36;
  }
  sub_7E57B0(v20, v21, a1, a2, a3, v22, &v40);
  if ( !v11 )
  {
    v23 = *(_QWORD *)(a4 + 184);
    *(_QWORD *)(*(_QWORD *)(v23 + 184) + 120LL) = v42;
    *(_QWORD *)(v23 + 184) = v43;
  }
  return (__int64)sub_729730(v41);
}
