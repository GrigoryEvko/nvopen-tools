// Function: sub_1181D90
// Address: 0x1181d90
//
__int64 __fastcall sub_1181D90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v5; // rbx
  int v6; // eax
  __int64 v7; // r12
  unsigned __int8 v8; // al
  _BYTE *v9; // rdx
  __int64 v10; // r13
  __int64 v11; // r15
  __int64 v13; // rax
  _BYTE *v14; // r14
  char v15; // al
  __int64 v16; // rsi
  _BYTE *v17; // rdx
  char v18; // al
  _BYTE *v19; // rax
  __int64 v20; // rdx
  int v21; // r13d
  unsigned int v22; // r14d
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 *v26; // rax
  __int64 v27; // rax
  char v28; // al
  _BYTE *v29; // rax
  __int64 *v30; // rax
  __int64 v31; // rdx
  __int64 *v32; // rax
  __int64 v33; // rcx
  __int64 v34; // rcx
  _BYTE *v35; // [rsp+8h] [rbp-68h]
  _BYTE *v36; // [rsp+10h] [rbp-60h]
  _BYTE *v37; // [rsp+10h] [rbp-60h]
  _QWORD *v39; // [rsp+28h] [rbp-48h] BYREF
  _QWORD *v40; // [rsp+30h] [rbp-40h] BYREF
  _BYTE *v41; // [rsp+38h] [rbp-38h]

  v4 = 0;
  v5 = a3;
  v6 = *(_WORD *)(a1 + 2) & 0x3F;
  if ( (unsigned int)(v6 - 32) > 1 )
    return v4;
  v7 = a2;
  if ( v6 == 33 )
  {
    v5 = a2;
    v7 = a3;
  }
  v8 = *(_BYTE *)v5;
  v9 = *(_BYTE **)(a1 - 64);
  v10 = *(_QWORD *)(a1 - 32);
  if ( *(_BYTE *)v5 > 0x1Cu && (v8 == 68 || v8 == 67) && (v11 = *(_QWORD *)(v5 - 32)) != 0 )
    v8 = *(_BYTE *)v11;
  else
    v11 = v5;
  if ( v8 != 85 )
    return 0;
  v13 = *(_QWORD *)(v11 - 32);
  if ( !v13 )
    return 0;
  if ( *(_BYTE *)v13
    || *(_QWORD *)(v13 + 24) != *(_QWORD *)(v11 + 80)
    || *(_DWORD *)(v13 + 36) != 67
    || (a2 = -32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF),
        (v14 = *(_BYTE **)(v11 - 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF))) == 0) )
  {
    if ( *(_BYTE *)v13 )
      return 0;
    if ( *(_QWORD *)(v13 + 24) != *(_QWORD *)(v11 + 80) )
      return 0;
    if ( *(_DWORD *)(v13 + 36) != 65 )
      return 0;
    v14 = *(_BYTE **)(v11 - 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF));
    if ( !v14 )
      return 0;
  }
  if ( v9 != v14 || (v36 = *(_BYTE **)(a1 - 64), v28 = sub_1178DE0(*(_QWORD *)(a1 - 32)), v9 = v36, !v28) )
  {
    v40 = 0;
    v41 = v9;
    if ( *v14 != 59 )
      return 0;
    v35 = v9;
    v15 = sub_995B10(&v40, *((_QWORD *)v14 - 8));
    v16 = *((_QWORD *)v14 - 4);
    v17 = v35;
    if ( v15 && (_BYTE *)v16 == v41 || (v18 = sub_995B10(&v40, v16), v17 = v35, v18) && *((_BYTE **)v14 - 8) == v41 )
    {
      a2 = v10;
      v37 = v17;
      v39 = 0;
      if ( (unsigned __int8)sub_995B10(&v39, v10) )
        goto LABEL_30;
      v17 = v37;
    }
    if ( *v14 != 59 )
      return 0;
    v19 = (_BYTE *)*((_QWORD *)v14 - 8);
    a2 = *((_QWORD *)v14 - 4);
    if ( (v17 != v19 || a2 != v10) && ((_BYTE *)v10 != v19 || (_BYTE *)a2 != v17) )
      return 0;
  }
LABEL_30:
  v21 = sub_BCB060(*(_QWORD *)(v11 + 8));
  if ( *(_BYTE *)v7 != 17 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17 > 1 )
      goto LABEL_34;
    if ( *(_BYTE *)v7 > 0x15u )
      goto LABEL_34;
    a2 = 0;
    v29 = sub_AD7630(v7, 0, v20);
    v7 = (__int64)v29;
    if ( !v29 || *v29 != 17 )
      goto LABEL_34;
  }
  v22 = *(_DWORD *)(v7 + 32);
  if ( v22 > 0x40 )
  {
    if ( v22 - (unsigned int)sub_C444A0(v7 + 24) > 0x40 )
      goto LABEL_34;
    v23 = **(_QWORD **)(v7 + 24);
  }
  else
  {
    v23 = *(_QWORD *)(v7 + 24);
  }
  if ( v21 != v23 )
  {
LABEL_34:
    v24 = *(_QWORD *)(v11 + 16);
    if ( v24 )
    {
      if ( !*(_QWORD *)(v24 + 8) )
      {
        v25 = *(_QWORD *)(v5 + 16);
        if ( v25 )
        {
          v4 = *(_QWORD *)(v25 + 8);
          if ( !v4 )
          {
            v40 = 0;
            if ( !(unsigned __int8)sub_993A50(&v40, *(_QWORD *)(v11 + 32 * (1LL - (*(_DWORD *)(v11 + 4) & 0x7FFFFFF)))) )
            {
              v26 = (__int64 *)sub_BD5C60(v11);
              v27 = sub_ACD6D0(v26);
              sub_AC2B30(v11 + 32 * (1LL - (*(_DWORD *)(v11 + 4) & 0x7FFFFFF)), v27);
              sub_B44E20((unsigned __int8 *)v11);
              sub_F15FC0(*(_QWORD *)(a4 + 40), v11);
              return v4;
            }
          }
        }
      }
    }
    return 0;
  }
  v30 = (__int64 *)sub_BD5C60(v11);
  v31 = sub_ACD720(v30);
  v32 = (__int64 *)(v11 + 32 * (1LL - (*(_DWORD *)(v11 + 4) & 0x7FFFFFF)));
  if ( *v32 )
  {
    a2 = v32[2];
    v33 = v32[1];
    *(_QWORD *)a2 = v33;
    if ( v33 )
    {
      a2 = v32[2];
      *(_QWORD *)(v33 + 16) = a2;
    }
  }
  *v32 = v31;
  if ( v31 )
  {
    v34 = *(_QWORD *)(v31 + 16);
    a2 = v31 + 16;
    v32[1] = v34;
    if ( v34 )
      *(_QWORD *)(v34 + 16) = v32 + 1;
    v32[2] = a2;
    *(_QWORD *)(v31 + 16) = v32;
  }
  v4 = v5;
  sub_B44F30((unsigned __int8 *)v11);
  sub_B44B50((__int64 *)v11, a2);
  sub_B44A60(v11);
  sub_F15FC0(*(_QWORD *)(a4 + 40), v11);
  return v4;
}
