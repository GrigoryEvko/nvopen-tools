// Function: sub_9A13D0
// Address: 0x9a13d0
//
__int16 __fastcall sub_9A13D0(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        unsigned __int8 *a4,
        int a5,
        unsigned __int8 a6,
        int a7)
{
  unsigned __int8 v7; // r15
  int *v12; // rax
  __int64 v13; // r8
  int v14; // eax
  __int64 v15; // rsi
  __int64 v16; // r10
  int v17; // edx
  __int64 v18; // rcx
  __int64 v19; // rdi
  __int16 result; // ax
  char v21; // al
  __int64 v22; // rdi
  unsigned int v23; // eax
  __int64 v24; // r10
  bool v25; // zf
  char v26; // dl
  char v27; // al
  __int64 v28; // rsi
  _BYTE *v29; // rdi
  __int64 v30; // r14
  __int64 v31; // r11
  __int16 v32; // ax
  __int16 v33; // ax
  char v34; // bl
  unsigned __int8 *v35; // rbx
  unsigned __int8 *v36; // rdx
  unsigned int v37; // r15d
  char v38; // dl
  __int16 v39; // ax
  unsigned int v40; // eax
  __int64 v41; // rax
  unsigned __int8 *v42; // rdi
  unsigned __int8 *v43; // rsi
  int v44; // eax
  __int64 v45; // rax
  _BYTE *v46; // rdi
  __int16 v47; // ax
  __int64 *v48; // r10
  char v49; // [rsp+Fh] [rbp-61h]
  __int64 v50; // [rsp+10h] [rbp-60h]
  unsigned __int8 *v51; // [rsp+10h] [rbp-60h]
  unsigned __int8 *v52; // [rsp+10h] [rbp-60h]
  char v53; // [rsp+18h] [rbp-58h]
  __int64 v54; // [rsp+18h] [rbp-58h]
  __int64 v55; // [rsp+18h] [rbp-58h]
  __int64 v56; // [rsp+18h] [rbp-58h]
  __int64 v58; // [rsp+28h] [rbp-48h] BYREF
  _QWORD *v59; // [rsp+30h] [rbp-40h] BYREF
  __int64 *v60; // [rsp+38h] [rbp-38h]

  v7 = a6;
  v58 = a1;
  v12 = (int *)sub_C94E20(qword_4F862D0);
  if ( v12 )
    v14 = *v12;
  else
    v14 = qword_4F862D0[2];
  if ( a7 == v14 )
    return 0;
  v15 = *(_QWORD *)(a3 + 8);
  v16 = v58;
  v17 = *(unsigned __int8 *)(v15 + 8);
  v18 = *(unsigned __int8 *)(*(_QWORD *)(v58 + 8) + 8LL);
  if ( (unsigned int)(v17 - 17) > 1 )
  {
    v19 = 0;
    if ( (_DWORD)v18 == 18 )
      return 0;
  }
  else
  {
    v19 = 1;
    if ( (_DWORD)v18 == 18 )
    {
      if ( *(_BYTE *)v58 == 83 && (_BYTE)qword_4F800E8 )
      {
LABEL_45:
        v35 = *(unsigned __int8 **)(v58 - 64);
        v36 = *(unsigned __int8 **)(v58 - 32);
        v37 = *(_WORD *)(v58 + 2) & 0x3F;
        if ( !a6 )
        {
          v19 = v37;
          v51 = *(unsigned __int8 **)(v58 - 32);
          v40 = sub_B52870(v37);
          v36 = v51;
          v37 = v40;
        }
        if ( (unsigned __int8 *)a3 != v35 )
          goto LABEL_48;
        if ( a4 == v36 )
        {
          v47 = sub_B53860(v37, (unsigned int)a2);
          v53 = v47;
          v38 = HIBYTE(v47);
          goto LABEL_50;
        }
        if ( *v36 == 18 && *a4 == 18 )
        {
          v52 = v36;
          v41 = sub_C33340(v19, v15, v36, v18, v13);
          v42 = v52 + 24;
          v43 = a4 + 24;
          if ( *((_QWORD *)v52 + 3) == v41 )
            v44 = sub_C3E510(v42, v43);
          else
            v44 = sub_C37950(v42, v43);
          if ( v37 - 4 <= 1 )
          {
            if ( ((unsigned int)(a2 - 4) <= 1 || (_DWORD)a2 == 12) && !v44 )
              goto LABEL_61;
          }
          else if ( v37 - 2 <= 1 && (unsigned int)(a2 - 2) <= 1 && v44 == 2 )
          {
LABEL_61:
            v53 = 1;
            v38 = 1;
            goto LABEL_50;
          }
        }
        else
        {
LABEL_48:
          if ( v37 == (_DWORD)a2 )
          {
            v39 = sub_9959F0(v37, v35, v36, (unsigned __int8 *)a3, a4);
            v53 = v39;
            v38 = HIBYTE(v39);
            goto LABEL_50;
          }
        }
        v38 = 0;
LABEL_50:
        LOBYTE(result) = v53;
        HIBYTE(result) = v38;
        return result;
      }
      goto LABEL_31;
    }
  }
  LOBYTE(v18) = (_DWORD)v18 == 17;
  if ( (_BYTE)v18 != (_BYTE)v19 )
    return 0;
  if ( *(_BYTE *)v58 == 83 && (_BYTE)qword_4F800E8 )
    goto LABEL_45;
  if ( (unsigned int)(v17 - 17) <= 1 )
LABEL_31:
    LOBYTE(v17) = *(_BYTE *)(**(_QWORD **)(v15 + 16) + 8LL);
  if ( (unsigned __int8)v17 <= 3u || (_BYTE)v17 == 5 || (v17 & 0xFD) == 4 )
    return 0;
  v59 = 0;
  v60 = &v58;
  if ( *(_BYTE *)v58 == 59 )
  {
    v50 = v58;
    v27 = sub_995B10(&v59, *(_QWORD *)(v58 - 64));
    v28 = *(_QWORD *)(v50 - 32);
    if ( v27 && v28 )
    {
      *v60 = v28;
    }
    else
    {
      if ( !(unsigned __int8)sub_995B10(&v59, v28) )
        goto LABEL_36;
      v45 = *(_QWORD *)(v50 - 64);
      if ( !v45 )
        goto LABEL_36;
      *v60 = v45;
    }
    v7 = a6 ^ 1;
LABEL_36:
    v16 = v58;
  }
  v21 = *(_BYTE *)v16;
  if ( *(_BYTE *)v16 <= 0x1Cu )
    return 0;
  if ( v21 != 82 )
  {
    if ( (unsigned __int8)(v21 - 57) > 1u && v21 != 86 )
      return 0;
    v22 = *(_QWORD *)(v16 + 8);
    v23 = *(unsigned __int8 *)(v22 + 8) - 17;
    if ( v7 )
    {
      if ( v23 <= 1 )
        v22 = **(_QWORD **)(v22 + 16);
      v54 = v16;
      if ( !(unsigned __int8)sub_BCAC40(v22, 1) )
        goto LABEL_28;
      v24 = v54;
      if ( *(_BYTE *)v54 != 57 )
      {
        if ( *(_BYTE *)v54 != 86 )
          goto LABEL_28;
        v25 = *(_QWORD *)(*(_QWORD *)(v54 - 96) + 8LL) == *(_QWORD *)(v54 + 8);
        v55 = *(_QWORD *)(v54 - 96);
        if ( !v25 )
          goto LABEL_28;
        v29 = *(_BYTE **)(v24 - 32);
        if ( *v29 > 0x15u )
          goto LABEL_28;
        v30 = *(_QWORD *)(v24 - 64);
        if ( !(unsigned __int8)sub_AC30F0(v29) )
          goto LABEL_28;
        goto LABEL_39;
      }
    }
    else
    {
      if ( v23 <= 1 )
        v22 = **(_QWORD **)(v22 + 16);
      v56 = v16;
      if ( !(unsigned __int8)sub_BCAC40(v22, 1) )
        goto LABEL_28;
      v24 = v56;
      if ( *(_BYTE *)v56 != 58 )
      {
        if ( *(_BYTE *)v56 != 86 )
          goto LABEL_28;
        v25 = *(_QWORD *)(*(_QWORD *)(v56 - 96) + 8LL) == *(_QWORD *)(v56 + 8);
        v55 = *(_QWORD *)(v56 - 96);
        if ( !v25 )
          goto LABEL_28;
        v46 = *(_BYTE **)(v24 - 64);
        if ( *v46 > 0x15u )
          goto LABEL_28;
        v30 = *(_QWORD *)(v24 - 32);
        if ( !(unsigned __int8)sub_AD7A80(v46) )
          goto LABEL_28;
LABEL_39:
        LODWORD(v31) = v55;
        if ( v30 )
          goto LABEL_40;
        goto LABEL_28;
      }
    }
    if ( (*(_BYTE *)(v24 + 7) & 0x40) != 0 )
      v48 = *(__int64 **)(v24 - 8);
    else
      v48 = (__int64 *)(v24 - 32LL * (*(_DWORD *)(v24 + 4) & 0x7FFFFFF));
    v31 = *v48;
    if ( *v48 )
    {
      v30 = v48[4];
      if ( v30 )
      {
LABEL_40:
        v32 = sub_9A13D0(v31, a2, a3, (_DWORD)a4, a5, v7, a7 + 1);
        v26 = HIBYTE(v32);
        v49 = v32;
        if ( !HIBYTE(v32) )
        {
          v33 = sub_9A13D0(v30, a2, a3, (_DWORD)a4, a5, v7, a7 + 1);
          v34 = v49;
          v26 = HIBYTE(v33);
          if ( HIBYTE(v33) )
            v34 = v33;
          v49 = v34;
        }
        goto LABEL_29;
      }
    }
LABEL_28:
    v26 = 0;
LABEL_29:
    LOBYTE(result) = v49;
    HIBYTE(result) = v26;
    return result;
  }
  return sub_9A0640(v16, a2, (unsigned __int8 *)a3, a4, v7);
}
