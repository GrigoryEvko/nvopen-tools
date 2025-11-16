// Function: sub_BD3110
// Address: 0xbd3110
//
__int64 __fastcall sub_BD3110(__int64 a1, unsigned int a2, __int64 a3)
{
  unsigned int v4; // r12d
  __int64 v5; // rdx
  unsigned __int8 **v6; // r15
  __int64 v7; // r8
  __int64 v8; // rdx
  unsigned __int8 **v9; // rbx
  unsigned __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // r8
  unsigned __int8 v13; // al
  unsigned __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // r11
  unsigned int v19; // r14d
  int v20; // eax
  bool v21; // al
  unsigned __int64 v22; // r14
  unsigned __int64 v23; // r10
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r11
  unsigned __int64 v30; // rax
  unsigned int v31; // esi
  _QWORD *v32; // rdi
  __int64 v33; // r8
  unsigned __int8 v34; // dl
  unsigned __int64 v36; // r10
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  _QWORD *v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rax
  unsigned __int64 v45; // [rsp+8h] [rbp-88h]
  __int64 v46; // [rsp+10h] [rbp-80h]
  unsigned __int64 v47; // [rsp+18h] [rbp-78h]
  __int64 v48; // [rsp+20h] [rbp-70h]
  __int64 v49; // [rsp+20h] [rbp-70h]
  char v50; // [rsp+20h] [rbp-70h]
  __int64 v51; // [rsp+20h] [rbp-70h]
  __int64 v52; // [rsp+20h] [rbp-70h]
  __int64 v53; // [rsp+20h] [rbp-70h]
  __int64 v54; // [rsp+20h] [rbp-70h]
  __int64 v55; // [rsp+28h] [rbp-68h]
  __int64 v56; // [rsp+30h] [rbp-60h]
  int v57; // [rsp+30h] [rbp-60h]
  __int64 v59; // [rsp+50h] [rbp-40h] BYREF
  __int64 v60; // [rsp+58h] [rbp-38h]

  v4 = a2;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v5 = *(_QWORD *)(a1 - 8);
  else
    v5 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  v56 = v5;
  v6 = (unsigned __int8 **)(v5 + 32);
  v7 = sub_BB5290(a1) & 0xFFFFFFFFFFFFFFF9LL | 4;
  if ( a2 != 1 )
  {
    v8 = v56;
    v9 = (unsigned __int8 **)(v56 + 32 * (a2 - 2 + 2LL));
    while ( 1 )
    {
      while ( 1 )
      {
        v10 = v7 & 0xFFFFFFFFFFFFFFF8LL;
        v11 = v7 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v7 )
          goto LABEL_39;
        v12 = (v7 >> 1) & 3;
        if ( v12 == 2 )
        {
          if ( !v10 )
            goto LABEL_39;
          goto LABEL_10;
        }
        if ( v12 == 1 && v10 )
        {
          v11 = *(_QWORD *)(v10 + 24);
          goto LABEL_10;
        }
LABEL_39:
        v11 = sub_BCBAE0(v11, *v6, v8);
LABEL_10:
        v13 = *(_BYTE *)(v11 + 8);
        if ( v13 != 16 )
          break;
        v7 = *(_QWORD *)(v11 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
LABEL_6:
        v6 += 4;
        if ( v6 == v9 )
          goto LABEL_13;
      }
      v14 = v11 & 0xFFFFFFFFFFFFFFF9LL;
      v8 = (unsigned int)v13 - 17;
      if ( (unsigned int)v8 > 1 )
      {
        v7 = v14;
        if ( v13 == 15 )
          goto LABEL_6;
        v6 += 4;
        v11 = 0;
        if ( v6 == v9 )
        {
          v7 = 0;
          goto LABEL_13;
        }
        goto LABEL_39;
      }
      v6 += 4;
      v7 = v14 | 2;
      if ( v6 == v9 )
        goto LABEL_13;
    }
  }
  v9 = v6;
LABEL_13:
  v15 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( a2 != (_DWORD)v15 )
  {
    v55 = 0;
    v16 = v7;
    v57 = v15 - 1;
    while ( 1 )
    {
      v17 = 32 * (v4 - v15);
      v18 = *(_QWORD *)(a1 + v17);
      if ( *(_BYTE *)v18 != 17 )
      {
LABEL_43:
        LOBYTE(v60) = 0;
        return v59;
      }
      v19 = *(_DWORD *)(v18 + 32);
      if ( v19 <= 0x40 )
      {
        v21 = *(_QWORD *)(v18 + 24) == 0;
      }
      else
      {
        v48 = *(_QWORD *)(a1 + v17);
        v20 = sub_C444A0(v18 + 24);
        v18 = v48;
        v21 = v19 == v20;
      }
      v22 = v16 & 0xFFFFFFFFFFFFFFF8LL;
      v23 = v16 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v21 )
        break;
LABEL_29:
      if ( !v16 )
        goto LABEL_32;
      v33 = (v16 >> 1) & 3;
      if ( v33 == 2 )
      {
        if ( !v22 )
LABEL_32:
          v23 = sub_BCBAE0(v22, *v9, v15);
        v34 = *(_BYTE *)(v23 + 8);
        if ( v34 == 16 )
          goto LABEL_34;
        goto LABEL_49;
      }
      if ( v33 != 1 || !v22 )
        goto LABEL_32;
      v23 = *(_QWORD *)(v22 + 24);
      v34 = *(_BYTE *)(v23 + 8);
      if ( v34 == 16 )
      {
LABEL_34:
        v16 = *(_QWORD *)(v23 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
        goto LABEL_35;
      }
LABEL_49:
      v36 = v23 & 0xFFFFFFFFFFFFFFF9LL;
      if ( (unsigned int)v34 - 17 > 1 )
      {
        v16 = 0;
        if ( v34 == 15 )
          v16 = v36;
      }
      else
      {
        v16 = v36 | 2;
      }
LABEL_35:
      v9 += 4;
      if ( v4 == v57 )
        goto LABEL_71;
      ++v4;
      v15 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
    }
    v24 = (v16 >> 1) & 3;
    if ( v16 )
    {
      if ( !v24 )
      {
        if ( v22 )
        {
          v53 = v18;
          v39 = sub_AE4AC0(a3, v16 & 0xFFFFFFFFFFFFFFF8LL);
          v40 = *(_QWORD **)(v53 + 24);
          if ( *(_DWORD *)(v53 + 32) > 0x40u )
            v40 = (_QWORD *)*v40;
          v41 = v39 + 16LL * (unsigned int)v40 + 24;
          v42 = *(_QWORD *)v41;
          LOBYTE(v41) = *(_BYTE *)(v41 + 8);
          v59 = v42;
          LOBYTE(v60) = v41;
          v43 = sub_CA1930(&v59);
          v23 = v16 & 0xFFFFFFFFFFFFFFF8LL;
          v55 += v43;
          goto LABEL_29;
        }
        goto LABEL_22;
      }
      if ( v24 == 2 )
      {
        v26 = v16 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v22 )
          goto LABEL_23;
        goto LABEL_22;
      }
      if ( v24 != 1 )
      {
LABEL_22:
        v49 = v18;
        v25 = sub_BCBAE0(v16 & 0xFFFFFFFFFFFFFFF8LL, *v9, v24);
        v18 = v49;
        v23 = v16 & 0xFFFFFFFFFFFFFFF8LL;
        v26 = v25;
LABEL_23:
        v45 = v23;
        v46 = v18;
        v50 = sub_AE5020(a3, v26);
        v27 = sub_9208B0(a3, v26);
        v23 = v45;
        v59 = v27;
        v29 = v46;
        v60 = v28;
        v30 = (((unsigned __int64)(v27 + 7) >> 3) + (1LL << v50) - 1) >> v50 << v50;
LABEL_24:
        if ( (_BYTE)v28 )
          goto LABEL_43;
        v31 = *(_DWORD *)(v29 + 32);
        v32 = *(_QWORD **)(v29 + 24);
        if ( v31 > 0x40 )
        {
          v15 = *v32 * v30;
          v55 += v15;
        }
        else
        {
          v15 = 0;
          if ( v31 )
            v15 = v30 * ((__int64)((_QWORD)v32 << (64 - (unsigned __int8)v31)) >> (64 - (unsigned __int8)v31));
          v55 += v15;
        }
        goto LABEL_29;
      }
      if ( v22 )
      {
        v26 = *(_QWORD *)(v22 + 24);
      }
      else
      {
        v54 = v18;
        v44 = sub_BCBAE0(0, *v9, 1);
        v18 = v54;
        v23 = v16 & 0xFFFFFFFFFFFFFFF8LL;
        v26 = v44;
      }
    }
    else
    {
      v52 = v18;
      v38 = sub_BCBAE0(v22, *v9, v24);
      v18 = v52;
      v23 = v16 & 0xFFFFFFFFFFFFFFF8LL;
      v26 = v38;
      if ( ((v16 >> 1) & 3) != 1 )
        goto LABEL_23;
    }
    v47 = v23;
    v51 = v18;
    v37 = sub_9208B0(a3, v26);
    v29 = v51;
    v23 = v47;
    v59 = v37;
    v60 = v28;
    v30 = (unsigned __int64)(v37 + 7) >> 3;
    goto LABEL_24;
  }
  v55 = 0;
LABEL_71:
  LOBYTE(v60) = 1;
  return v55;
}
