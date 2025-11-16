// Function: sub_8C0230
// Address: 0x8c0230
//
_QWORD *__fastcall sub_8C0230(__int64 a1, __m128i **a2, int a3, unsigned int a4, int a5)
{
  __int64 v6; // rbx
  char v7; // al
  _QWORD *v8; // r13
  __int64 v9; // rcx
  _UNKNOWN *__ptr32 *v10; // r8
  __int64 v11; // r9
  __m128i *v12; // r15
  char v13; // r12
  unsigned __int8 *v14; // rdi
  _QWORD **v15; // rax
  _QWORD *v16; // r14
  __int64 v17; // r8
  __int64 v18; // r9
  char v19; // al
  __int64 v20; // r12
  bool v21; // r12
  _QWORD *v22; // rax
  char v23; // al
  __int16 v24; // r12
  __int16 v25; // r12
  char v26; // al
  __int64 v27; // rdi
  _QWORD *v28; // rax
  char v29; // dl
  __int64 v30; // rcx
  __m128i *v31; // rdi
  __int64 v33; // rdx
  _QWORD *v34; // r15
  __int64 v35; // rdi
  __int64 v36; // rax
  int v39; // [rsp+14h] [rbp-7Ch]
  int v40; // [rsp+28h] [rbp-68h] BYREF
  int v41; // [rsp+2Ch] [rbp-64h] BYREF
  __m128i *v42; // [rsp+30h] [rbp-60h] BYREF
  __m128i *v43; // [rsp+38h] [rbp-58h] BYREF
  _QWORD v44[2]; // [rsp+40h] [rbp-50h] BYREF
  int v45; // [rsp+50h] [rbp-40h]

  v6 = a1;
  v7 = *(_BYTE *)(a1 + 80);
  if ( v7 == 16 )
  {
    v6 = **(_QWORD **)(a1 + 88);
    v7 = *(_BYTE *)(v6 + 80);
  }
  if ( v7 == 24 )
    v6 = *(_QWORD *)(v6 + 88);
  v8 = *(_QWORD **)(v6 + 88);
  sub_89A380(0, *a2, &v42, &v43, &v41, &v40);
  v39 = v40;
  if ( !v40 && ((*(_BYTE *)(v6 + 81) & 0x10) == 0 || (*(_BYTE *)(*(_QWORD *)(v6 + 64) + 177LL) & 0x20) == 0) )
  {
    if ( !v8[13] || dword_4F04C44 != -1 )
    {
      v12 = v43;
      v13 = 1;
      goto LABEL_10;
    }
    v21 = 0;
    goto LABEL_56;
  }
  v39 = 1;
  v21 = a3 != 0;
  if ( v8[13] && dword_4F04C44 == -1 )
  {
LABEL_56:
    if ( (unsigned int)sub_825090() )
    {
      sub_8250A0();
    }
    else if ( (unsigned int)sub_8250B0() )
    {
      sub_8250C0();
    }
  }
  v12 = v43;
  if ( v21 )
  {
    v22 = v8;
    if ( *(_BYTE *)(v6 + 80) == 19 )
    {
      v33 = v8[25];
      if ( v33 )
        v22 = *(_QWORD **)(v33 + 88);
    }
    v23 = *((_BYTE *)v22 + 160);
    v16 = (_QWORD *)v8[18];
    v24 = 2 * ((v23 & 6) != 0);
    if ( (v23 & 0x10) != 0 )
      v24 = (2 * ((v23 & 6) != 0)) | 0x20;
    v25 = v24 | 8;
    if ( v16 )
    {
      while ( 1 )
      {
        v26 = *((_BYTE *)v16 + 80);
        if ( v26 == 19 )
        {
          v27 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v16[11] + 176LL) + 88LL) + 168LL) + 168LL);
        }
        else
        {
          if ( v26 != 21 )
            sub_721090();
          v27 = **(_QWORD **)(*(_QWORD *)(v16[11] + 192LL) + 216LL);
        }
        if ( sub_89AB40(v27, (__int64)v12, v25, v9, v10)
          && (*((_BYTE *)v16 + 80) != 19 || sub_890680(*(_QWORD *)(v16[11] + 176LL), 0)) )
        {
          break;
        }
        v16 = (_QWORD *)v16[1];
        if ( !v16 )
        {
          v12 = v43;
          v13 = (v39 ^ 1) & 1;
          goto LABEL_10;
        }
      }
LABEL_34:
      v28 = (_QWORD *)sub_892240((__int64)v16);
      v29 = *((_BYTE *)v16 + 80);
      if ( v29 == 9 || v29 == 7 )
      {
        v30 = v16[11];
      }
      else
      {
        v30 = 0;
        if ( v29 == 21 )
        {
          v30 = *(_QWORD *)(v16[11] + 192LL);
          if ( v39 )
          {
LABEL_38:
            if ( v41 )
              sub_725130(v42->m128i_i64);
            v31 = *a2;
LABEL_41:
            sub_725130(v31->m128i_i64);
            goto LABEL_42;
          }
LABEL_47:
          if ( (*((_BYTE *)v16 + 81) & 2) == 0 && *(char *)(v30 + 170) >= 0 && (*(_BYTE *)(v28[2] + 28LL) & 1) == 0 )
            sub_8AA320(v28, 0, a4);
          goto LABEL_38;
        }
      }
      if ( v39 )
        goto LABEL_38;
      goto LABEL_47;
    }
  }
  v13 = (v39 ^ 1) & 1;
LABEL_10:
  v14 = (unsigned __int8 *)v8[17];
  v44[0] = v6;
  v44[1] = v12;
  v45 = 0;
  if ( !v14 )
  {
    v36 = sub_881A70(0, 0xBu, 12, 13, (__int64)v10, v11);
    v8[17] = v36;
    v14 = (unsigned __int8 *)v36;
  }
  v15 = (_QWORD **)sub_881B20(v14, (__int64)v44, 0);
  if ( v15 )
  {
    v16 = *v15;
    if ( *v15 )
      goto LABEL_34;
    if ( !v13 )
      goto LABEL_16;
  }
  else if ( !v13 )
  {
    goto LABEL_16;
  }
  if ( !(unsigned int)sub_8A00C0(v6, v43->m128i_i64, a5) )
  {
    if ( v41 )
      sub_725130(v42->m128i_i64);
    v16 = 0;
    sub_725130((*a2)->m128i_i64);
    goto LABEL_42;
  }
LABEL_16:
  v16 = sub_88E8B0(v6, (__int64)v43);
  v19 = *((_BYTE *)v16 + 80);
  if ( v19 == 9 || v19 == 7 )
  {
    v20 = v16[11];
  }
  else
  {
    if ( v19 != 21 )
    {
      *(_QWORD *)(v16[12] + 48LL) = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C34 + 224);
      BUG();
    }
    v20 = *(_QWORD *)(v16[11] + 192LL);
  }
  v34 = (_QWORD *)v16[12];
  v35 = v34[4];
  v34[6] = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C34 + 224);
  sub_890140(v35, v8, (__int64)v16, **(_QWORD **)(v20 + 216), v17, v18);
  if ( v39 )
  {
    *(_BYTE *)(v20 + 170) |= 0x40u;
    *(_QWORD *)(v20 + 120) = dword_4D03B80;
    sub_735E40(v20, -1);
  }
  else
  {
    sub_8AA320(v34, 1, a4);
    sub_8AD0D0((__int64)v16, a4, 0);
  }
  sub_8CCE20(v16, v8);
  v31 = v42;
  if ( v43 != v42 )
    goto LABEL_41;
LABEL_42:
  *a2 = 0;
  return v16;
}
