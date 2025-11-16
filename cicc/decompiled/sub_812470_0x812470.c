// Function: sub_812470
// Address: 0x812470
//
__int64 __fastcall sub_812470(unsigned int *a1, _DWORD *a2, unsigned __int64 a3, _QWORD *a4)
{
  __int64 result; // rax
  unsigned int v8; // edx
  __int64 v9; // r12
  char v10; // r14
  _QWORD *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdi
  char v21; // al
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // rax
  char i; // dl
  __int64 v26; // rax
  int v27; // eax
  unsigned __int64 v28; // [rsp+0h] [rbp-60h]
  unsigned __int64 v29; // [rsp+0h] [rbp-60h]
  unsigned __int64 v30; // [rsp+8h] [rbp-58h]
  unsigned int v31; // [rsp+8h] [rbp-58h]
  int v32; // [rsp+8h] [rbp-58h]
  _DWORD v33[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v34; // [rsp+18h] [rbp-48h]
  char v35; // [rsp+20h] [rbp-40h]

  result = *a1;
  v8 = a1[1];
  v9 = *((_QWORD *)a1 + 1);
  v33[0] = result;
  v33[1] = v8;
  if ( !(_DWORD)result )
  {
    v13 = *(_QWORD *)(v9 + 40);
    v10 = *((_BYTE *)a1 + 16);
    if ( (*(_BYTE *)(v9 + 89) & 4) != 0 )
    {
      v14 = *(_QWORD *)(v13 + 32);
      v35 = 6;
      v34 = v14;
    }
    else
    {
      if ( !v13 || *(_BYTE *)(v13 + 28) != 3 )
      {
        v34 = 0;
        v35 = 0;
        if ( v10 != 6 )
        {
          if ( !v8 || (LODWORD(result) = dword_4D0425C) != 0 )
          {
            *a4 += 2LL;
            sub_8238B0(qword_4F18BE0, "sr", 2);
            goto LABEL_31;
          }
          goto LABEL_44;
        }
LABEL_37:
        v30 = a3;
        v19 = sub_8D2220(v9);
        a3 = v30;
        v20 = v19;
        v21 = *(_BYTE *)(v19 + 140);
        if ( v21 == 12 )
        {
          if ( *(_BYTE *)(v20 + 184) == 1 )
          {
LABEL_40:
            v8 = a1[1];
            v10 = 6;
            result = 1;
            goto LABEL_5;
          }
        }
        else if ( (unsigned __int8)(v21 - 9) <= 2u && *(_QWORD *)(*(_QWORD *)(v20 + 168) + 256LL) )
        {
          goto LABEL_40;
        }
        v10 = 6;
        v27 = sub_8D3D40(v20);
        v8 = a1[1];
        a3 = v30;
        result = v27 != 0;
        goto LABEL_5;
      }
      v18 = *(_QWORD *)(v13 + 32);
      v35 = 28;
      v34 = v18;
    }
    sub_812470(v33, a2, a3 + 1, a4);
    if ( v10 != 6 )
    {
      if ( v10 == 28 )
        goto LABEL_32;
      goto LABEL_19;
    }
LABEL_49:
    v24 = v9;
    goto LABEL_50;
  }
  if ( !v9 )
  {
    v34 = 0;
    if ( !v8 )
      return result;
    v10 = 0;
    result = 0;
    goto LABEL_6;
  }
  result = *(_QWORD *)(v9 + 16);
  v34 = result;
  if ( (*(_BYTE *)(v9 + 32) & 1) != 0 )
  {
    v9 = *(_QWORD *)(v9 + 8);
    if ( !result )
      goto LABEL_37;
    result = sub_812470(v33, a2, a3 + 1, a4);
    if ( !v9 )
      return result;
    goto LABEL_49;
  }
  v9 = *(_QWORD *)(v9 + 8);
  v10 = 28;
  if ( result )
  {
    result = sub_812470(v33, a2, a3 + 1, a4);
    if ( !v9 )
      return result;
LABEL_32:
    v15 = *(_QWORD *)(v9 + 8);
    v16 = v15;
    if ( (*(_BYTE *)(v9 + 89) & 8) != 0 )
      v16 = *(_QWORD *)(v9 + 24);
    if ( v16 )
    {
      v16 = 0;
    }
    else
    {
      if ( v15 )
        return (__int64)sub_812380(v9, v16, 0, a4);
      sub_80B070(v9, (__int64)a4);
      v15 = *(_QWORD *)(v9 + 8);
    }
LABEL_20:
    if ( *a1 )
    {
      v17 = *((_QWORD *)a1 + 1);
      if ( v17 )
      {
        if ( !v15 )
        {
          v26 = *(_QWORD *)(v17 + 24);
          if ( v26 )
          {
            *(_QWORD *)(v9 + 8) = v26;
            result = (__int64)sub_812380(v9, v16, 0, a4);
            *(_QWORD *)(v9 + 8) = 0;
            return result;
          }
        }
      }
    }
    return (__int64)sub_812380(v9, v16, 0, a4);
  }
LABEL_5:
  if ( !v8 )
    goto LABEL_7;
LABEL_6:
  if ( !dword_4D0425C )
  {
LABEL_44:
    v22 = qword_4F18BE0;
    *a4 += 2LL;
    v28 = a3;
    v31 = result;
    sub_8238B0(v22, "gs", 2);
    a3 = v28;
    result = v31;
    if ( !v9 )
      goto LABEL_8;
    goto LABEL_45;
  }
LABEL_7:
  if ( !v9 )
  {
LABEL_8:
    if ( v10 != 6 )
      return result;
    result = dword_4D0425C | (unsigned int)result;
    if ( !(_DWORD)result )
      return result;
    goto LABEL_10;
  }
LABEL_45:
  v23 = qword_4F18BE0;
  *a4 += 2LL;
  v29 = a3;
  v32 = result;
  sub_8238B0(v23, "sr", 2);
  if ( v10 != 6 )
  {
LABEL_31:
    *a2 = 1;
    if ( v10 == 28 )
      goto LABEL_32;
LABEL_19:
    v15 = *(_QWORD *)(v9 + 8);
    v16 = 0;
    goto LABEL_20;
  }
  a3 = v29;
  if ( !(dword_4D0425C | v32) )
  {
    *a2 = 1;
    v24 = v9;
LABEL_50:
    for ( i = *(_BYTE *)(v9 + 140); i == 12; i = *(_BYTE *)(v24 + 140) )
      v24 = *(_QWORD *)(v24 + 160);
    v9 = v24;
    v16 = 0;
    v15 = *(_QWORD *)(v24 + 8);
    if ( (unsigned __int8)(i - 9) <= 2u )
      v16 = *(_QWORD *)(*(_QWORD *)(v9 + 168) + 168LL);
    goto LABEL_20;
  }
LABEL_10:
  if ( a3 > 1 )
  {
    v11 = (_QWORD *)qword_4F18BE0;
    ++*a4;
    v12 = v11[2];
    if ( (unsigned __int64)(v12 + 1) > v11[1] )
    {
      sub_823810(v11);
      v11 = (_QWORD *)qword_4F18BE0;
      v12 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v11[4] + v12) = 78;
    ++v11[2];
    *a2 = 1;
  }
  return sub_80F5E0(v9, 0, a4);
}
