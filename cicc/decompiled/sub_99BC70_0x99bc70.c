// Function: sub_99BC70
// Address: 0x99bc70
//
char __fastcall sub_99BC70(unsigned __int8 *a1, _BYTE *a2, __int64 *a3)
{
  bool v6; // zf
  char v7; // al
  unsigned __int8 v8; // dl
  unsigned __int8 *v9; // r15
  unsigned __int8 *v10; // r9
  char result; // al
  unsigned __int8 *v12; // rdx
  unsigned __int8 *v13; // rcx
  unsigned __int8 *v14; // rsi
  unsigned __int8 *v15; // r14
  _BYTE *v16; // r12
  __int64 v17; // rdi
  __int64 v18; // r13
  __int64 v19; // rax
  unsigned int v20; // eax
  unsigned __int8 *v21; // r14
  _BYTE *v22; // r14
  __int64 v23; // rdi
  __int64 v24; // rax
  unsigned __int8 *v25; // rsi
  char v26; // al
  _BYTE *v27; // rsi
  unsigned __int8 *v28; // r9
  char v29; // al
  _BYTE *v30; // rax
  _BYTE *v31; // rax
  _BYTE *v32; // rax
  unsigned __int8 **v33; // rdx
  unsigned __int8 **v34; // rcx
  unsigned __int8 **v35; // rcx
  unsigned __int8 **v36; // rax
  unsigned __int8 *v37; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v38; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v39; // [rsp+18h] [rbp-58h] BYREF
  _QWORD *v40; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int8 **v41; // [rsp+28h] [rbp-48h]
  _QWORD v42[8]; // [rsp+30h] [rbp-40h] BYREF

  v6 = *a1 == 57;
  v40 = 0;
  v41 = &v39;
  if ( v6
    && ((unsigned __int8)sub_996420(&v40, 30, *((unsigned __int8 **)a1 - 8))
     || (unsigned __int8)sub_996420(&v40, 30, *((unsigned __int8 **)a1 - 4))) )
  {
    v7 = *a2;
    if ( *a2 != 57 )
    {
LABEL_3:
      if ( v7 != 59 )
        goto LABEL_4;
LABEL_18:
      v12 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
      v13 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
      if ( *v12 == 57 )
      {
        v25 = (unsigned __int8 *)*((_QWORD *)v12 - 8);
        v15 = (unsigned __int8 *)*((_QWORD *)v12 - 4);
        if ( a1 == v25 && v25 )
        {
          if ( !v15 )
            goto LABEL_19;
        }
        else
        {
          if ( a1 != v15 || v15 == 0 || !v25 )
            goto LABEL_19;
          v15 = (unsigned __int8 *)*((_QWORD *)v12 - 8);
        }
        if ( v13 == v15 )
        {
LABEL_58:
          if ( sub_98EF80(a1, a3[4], a3[5], a3[3], 0) )
          {
            result = sub_98EF80(v15, a3[4], a3[5], a3[3], 0);
            if ( result )
              return result;
          }
          v7 = *a2;
          goto LABEL_4;
        }
      }
LABEL_19:
      if ( *v13 != 57 )
        goto LABEL_4;
      v14 = (unsigned __int8 *)*((_QWORD *)v13 - 8);
      v15 = (unsigned __int8 *)*((_QWORD *)v13 - 4);
      if ( a1 == v14 && v14 )
      {
        if ( !v15 )
          goto LABEL_4;
      }
      else
      {
        if ( v15 == 0 || a1 != v15 || !v14 )
          goto LABEL_4;
        v15 = (unsigned __int8 *)*((_QWORD *)v13 - 8);
      }
      if ( v12 != v15 )
        goto LABEL_4;
      goto LABEL_58;
    }
    if ( v39 != *((unsigned __int8 **)a2 - 8) && v39 != *((unsigned __int8 **)a2 - 4) )
    {
      v40 = 0;
      v41 = (unsigned __int8 **)a1;
      goto LABEL_15;
    }
    result = sub_98EF80(v39, a3[4], a3[5], a3[3], 0);
    if ( result )
      return result;
  }
  v7 = *a2;
  v40 = 0;
  v41 = (unsigned __int8 **)a1;
  if ( v7 != 57 )
    goto LABEL_3;
LABEL_15:
  if ( sub_9987C0((__int64)&v40, 30, *((unsigned __int8 **)a2 - 8))
    || sub_9987C0((__int64)&v40, 30, *((unsigned __int8 **)a2 - 4)) )
  {
    result = sub_98EF80(a1, a3[4], a3[5], a3[3], 0);
    if ( result )
      return result;
  }
  v7 = *a2;
  if ( *a2 == 59 )
    goto LABEL_18;
LABEL_4:
  v8 = *a1;
  if ( *a1 <= 0x1Cu )
    goto LABEL_11;
  if ( v8 != 68 && v8 != 69 )
    goto LABEL_7;
  v21 = (unsigned __int8 *)*((_QWORD *)a1 - 4);
  if ( !v21 )
    goto LABEL_11;
  v40 = 0;
  v41 = (unsigned __int8 **)v21;
  v42[0] = 0;
  v42[1] = v21;
  if ( v7 == 68 )
  {
    if ( sub_9987C0((__int64)&v40, 30, *((unsigned __int8 **)a2 - 4)) )
    {
LABEL_61:
      result = sub_98EF80(v21, a3[4], a3[5], a3[3], 0);
      if ( result )
        return result;
      goto LABEL_62;
    }
    v7 = *a2;
  }
  if ( v7 != 69 )
  {
    v8 = *a1;
    goto LABEL_7;
  }
  if ( sub_9987C0((__int64)v42, 30, *((unsigned __int8 **)a2 - 4)) )
    goto LABEL_61;
LABEL_62:
  v8 = *a1;
  v7 = *a2;
LABEL_7:
  if ( v8 != 57
    || (v9 = (unsigned __int8 *)*((_QWORD *)a1 - 8)) == 0
    || (v10 = (unsigned __int8 *)*((_QWORD *)a1 - 4)) == 0
    || (v40 = 0, v41 = (unsigned __int8 **)v9, v42[0] = v10, v7 != 59) )
  {
LABEL_11:
    if ( v7 != 54 )
      goto LABEL_12;
    v22 = (_BYTE *)*((_QWORD *)a2 - 4);
    if ( *v22 != 44 )
      return 0;
    v23 = *((_QWORD *)v22 - 8);
    v18 = v23 + 24;
    if ( *(_BYTE *)v23 != 17 )
    {
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v23 + 8) + 8LL) - 17 > 1 || *(_BYTE *)v23 > 0x15u )
        return 0;
      v31 = (_BYTE *)sub_AD7630(v23, 0);
      if ( !v31 || *v31 != 17 )
        goto LABEL_49;
      v18 = (__int64)(v31 + 24);
    }
    v24 = *((_QWORD *)v22 - 4);
    if ( v24 && *a1 == 55 && v24 == *((_QWORD *)a1 - 4) )
    {
LABEL_32:
      v20 = sub_BCB060(*((_QWORD *)a1 + 1));
      return !sub_986EE0(v18, v20);
    }
LABEL_49:
    v7 = *a2;
LABEL_12:
    if ( v7 != 55 )
      return 0;
    v16 = (_BYTE *)*((_QWORD *)a2 - 4);
    if ( *v16 != 44 )
      return 0;
    v17 = *((_QWORD *)v16 - 8);
    v18 = v17 + 24;
    if ( *(_BYTE *)v17 != 17 )
    {
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v17 + 8) + 8LL) - 17 > 1 )
        return 0;
      if ( *(_BYTE *)v17 > 0x15u )
        return 0;
      v32 = (_BYTE *)sub_AD7630(v17, 0);
      if ( !v32 || *v32 != 17 )
        return 0;
      v18 = (__int64)(v32 + 24);
    }
    v19 = *((_QWORD *)v16 - 4);
    if ( v19 && *a1 == 54 && v19 == *((_QWORD *)a1 - 4) )
      goto LABEL_32;
    return 0;
  }
  v37 = v10;
  v26 = sub_995B10(&v40, *((_QWORD *)a2 - 8));
  v27 = (_BYTE *)*((_QWORD *)a2 - 4);
  v28 = v37;
  if ( !v26
    || *v27 != 58
    || ((v33 = (unsigned __int8 **)*((_QWORD *)v27 - 8), v34 = (unsigned __int8 **)*((_QWORD *)v27 - 4), v33 != v41)
     || v34 != (unsigned __int8 **)v42[0])
    && (v34 != v41 || v33 != (unsigned __int8 **)v42[0]) )
  {
    v29 = sub_995B10(&v40, (__int64)v27);
    v28 = v37;
    if ( !v29 )
      goto LABEL_67;
    v30 = (_BYTE *)*((_QWORD *)a2 - 8);
    if ( *v30 != 58 )
      goto LABEL_67;
    v35 = (unsigned __int8 **)*((_QWORD *)v30 - 8);
    v36 = (unsigned __int8 **)*((_QWORD *)v30 - 4);
    if ( (v35 != v41 || v36 != (unsigned __int8 **)v42[0]) && (v36 != v41 || v35 != (unsigned __int8 **)v42[0]) )
      goto LABEL_67;
  }
  v38 = v28;
  if ( !sub_98EF80(v9, a3[4], a3[5], a3[3], 0) || (result = sub_98EF80(v38, a3[4], a3[5], a3[3], 0)) == 0 )
  {
LABEL_67:
    v7 = *a2;
    goto LABEL_11;
  }
  return result;
}
