// Function: sub_7D4600
// Address: 0x7d4600
//
__int64 __fastcall sub_7D4600(__int64 a1, __int64 *a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rdx
  __int64 *v9; // rax
  __int64 v10; // r14
  char v11; // al
  __int64 v13; // rsi
  char v14; // al
  __int64 v15; // r12
  __int64 v16; // r15
  char v17; // al
  __int64 v18; // rcx
  __int64 v19; // rax
  char v20; // al
  char v21; // al
  __int64 v22; // r15
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // [rsp+0h] [rbp-90h]
  __int64 v26; // [rsp+8h] [rbp-88h]
  int v28; // [rsp+24h] [rbp-6Ch] BYREF
  __int64 v29; // [rsp+28h] [rbp-68h] BYREF
  int v30; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v31; // [rsp+34h] [rbp-5Ch]
  int v32; // [rsp+38h] [rbp-58h]
  int v33; // [rsp+3Ch] [rbp-54h]
  int v34; // [rsp+40h] [rbp-50h]
  int v35; // [rsp+44h] [rbp-4Ch]
  _BOOL4 v36; // [rsp+48h] [rbp-48h]
  int v37; // [rsp+4Ch] [rbp-44h]
  int v38; // [rsp+50h] [rbp-40h]

  v30 = a3 & 1;
  v29 = 0;
  v31 = (a3 >> 1) & 1;
  v28 = 0;
  v32 = (a3 >> 11) & 1;
  v33 = (a3 >> 14) & 1;
  v34 = (a3 >> 5) & 1;
  v35 = (a3 >> 19) & 1;
  v36 = (a3 & 0x204020) == 0;
  v8 = 0;
  v37 = (a3 >> 13) & 1;
  if ( unk_4F04C48 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0 && dword_4D047C8 )
  {
    v8 = (unsigned int)sub_7D3BE0(a1, dword_4D047C8, 0, qword_4F04C68, a5);
    v9 = a2;
    v38 = v8;
    v10 = a2[3];
    if ( v10 )
      goto LABEL_4;
  }
  else
  {
    v9 = a2;
    v38 = 0;
    v10 = a2[3];
    if ( v10 )
      goto LABEL_4;
  }
  v13 = *v9;
  if ( unk_4D03F98 )
  {
    if ( a1 )
    {
      if ( *(_QWORD *)(v13 + 64) )
      {
        v14 = *(_BYTE *)(a1 + 28);
        if ( v14 == 3 || !v14 )
        {
          sub_824D70(a1);
          v13 = *a2;
        }
      }
    }
  }
  v15 = *(_QWORD *)(v13 + 24);
  if ( v15 )
  {
    v16 = 0;
    do
    {
      v17 = *(_BYTE *)(v15 + 80);
      v18 = v15;
      if ( v17 == 16 )
      {
        v18 = **(_QWORD **)(v15 + 88);
        v17 = *(_BYTE *)(v18 + 80);
      }
      if ( v17 == 24 )
        v18 = *(_QWORD *)(v18 + 88);
      if ( v16 && *(_DWORD *)(v16 + 40) != *(_DWORD *)(v15 + 40) )
      {
        v15 = v16;
        goto LABEL_29;
      }
      v13 = a1;
      if ( sub_7CEDF0(&v30, a1, v15, v18) )
      {
        v8 = v31;
        if ( !v31 || *(_BYTE *)(v15 + 80) != 3 )
          goto LABEL_29;
        v16 = v15;
      }
      v15 = *(_QWORD *)(v15 + 8);
    }
    while ( v15 );
    v15 = v16;
    if ( !v16 )
      goto LABEL_37;
LABEL_29:
    if ( (!v34 || v37) && !v35 )
    {
      v23 = sub_7D4400(a2, 0, *(__int64 **)(a1 + 184), a3, 0, &v29, &v28, 1);
      if ( v23 )
      {
        v15 = sub_7D09E0(v23, v15, (__int64)a2, 1u, 0, a3, &v28);
        if ( !v15 )
        {
LABEL_54:
          if ( v34 )
          {
LABEL_56:
            a2[3] = 0;
            return v10;
          }
          goto LABEL_55;
        }
      }
    }
LABEL_32:
    v10 = v15;
    a2[3] = v15;
    goto LABEL_4;
  }
LABEL_37:
  v19 = sub_880F60(*(unsigned int *)(a1 + 24), v13, v8);
  v25 = 0;
  v26 = 0;
  v15 = sub_883800(v19 + 24, *a2);
  if ( v15 )
  {
    do
    {
      v21 = *(_BYTE *)(v15 + 80);
      v22 = v15;
      if ( v21 == 16 )
      {
        v22 = **(_QWORD **)(v15 + 88);
        v21 = *(_BYTE *)(v22 + 80);
      }
      if ( v21 == 24 )
        v22 = *(_QWORD *)(v22 + 88);
      if ( sub_7CEDF0(&v30, a1, v15, v22) )
      {
        if ( v31 )
        {
          if ( *(_BYTE *)(v15 + 80) != 3 )
            goto LABEL_29;
          v26 = v15;
        }
        else
        {
          v20 = *(_BYTE *)(v22 + 80);
          if ( (unsigned __int8)(v20 - 4) > 2u && (v20 != 3 || !*(_BYTE *)(v22 + 104)) )
            goto LABEL_29;
          v25 = v15;
        }
      }
      v15 = *(_QWORD *)(v15 + 32);
    }
    while ( v15 );
    v15 = v26;
    if ( v26 )
      goto LABEL_29;
    v15 = v25;
    if ( v25 )
      goto LABEL_29;
  }
  if ( v34 )
  {
    if ( !v37 || v35 )
      goto LABEL_56;
LABEL_63:
    v15 = sub_7D4400(a2, 0, *(__int64 **)(a1 + 184), a3, 0, &v29, &v28, 1);
    if ( !v15 )
      goto LABEL_54;
    goto LABEL_32;
  }
  if ( !v35 )
    goto LABEL_63;
LABEL_55:
  if ( v35 | v33 )
    goto LABEL_56;
  v24 = sub_7D4400(a2, 0, *(__int64 **)(a1 + 184), a3, 0, &v29, &v28, 0);
  a2[3] = v24;
  v10 = v24;
  if ( !v24 )
    return v10;
LABEL_4:
  v11 = *(_BYTE *)(v10 + 80);
  if ( v11 == 16 )
  {
    v10 = **(_QWORD **)(v10 + 88);
    if ( *(_BYTE *)(v10 + 80) != 24 )
      return v10;
    return *(_QWORD *)(v10 + 88);
  }
  if ( v11 == 24 )
    return *(_QWORD *)(v10 + 88);
  return v10;
}
