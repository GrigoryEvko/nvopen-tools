// Function: sub_883C30
// Address: 0x883c30
//
__int64 __fastcall sub_883C30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v8; // r12
  __int64 v9; // rbx
  char v10; // al
  __int64 *v11; // rdx
  __int64 v12; // r8
  char v13; // r15
  int v14; // r11d
  int v15; // r10d
  __int64 v16; // rsi
  __int64 *v17; // rax
  unsigned int v18; // r15d
  __int64 v19; // r14
  int v20; // r13d
  __int64 v21; // rax
  __int64 v22; // rdx
  char v23; // cl
  __int64 v24; // rcx
  _QWORD *v25; // r9
  __int64 v26; // rsi
  int v27; // eax
  int v29; // eax
  __int64 *v30; // rax
  __int64 v31; // rax
  __int64 *v32; // rax
  __int64 v33; // rdi
  int v34; // eax
  int v35; // [rsp+Ch] [rbp-94h]
  __int64 v36; // [rsp+18h] [rbp-88h]
  char v37; // [rsp+20h] [rbp-80h]
  int v38; // [rsp+28h] [rbp-78h]
  __int64 v39; // [rsp+28h] [rbp-78h]
  __int64 v40; // [rsp+30h] [rbp-70h]
  __int64 v41; // [rsp+38h] [rbp-68h]
  int v43; // [rsp+40h] [rbp-60h]
  int v45; // [rsp+48h] [rbp-58h]
  _QWORD v46[10]; // [rsp+50h] [rbp-50h] BYREF

  v8 = (_QWORD *)a6;
  v9 = a4;
  v10 = sub_883850(a1, a2, a3, a4, a5, a6);
  if ( !v10 )
    return 1;
  v11 = (__int64 *)a3;
  v12 = a5;
  v13 = v10;
  if ( v10 == 3 )
  {
    v14 = 0;
    v15 = 0;
    goto LABEL_4;
  }
  v15 = sub_87D890(a2);
  if ( v15 )
    return 1;
  v11 = (__int64 *)a3;
  v12 = a5;
  if ( v13 == 1 )
  {
    if ( (unsigned int)sub_87D970(a2) )
      return 1;
    v11 = (__int64 *)a3;
    v12 = a5;
    v14 = 1;
    v15 = 1;
  }
  else
  {
    v14 = 1;
  }
LABEL_4:
  if ( *(_BYTE *)(v12 + 80) != 16 )
    goto LABEL_5;
  if ( (*(_BYTE *)(v12 + 96) & 4) == 0 || dword_4D04964 )
    goto LABEL_50;
  v18 = dword_4F077BC;
  if ( !dword_4F077BC )
    return v18;
  if ( qword_4F077A8 > 0x76BFu )
    return 0;
LABEL_50:
  if ( *(_BYTE *)(a1 + 80) == 16 && a1 == v12 )
    return 0;
LABEL_5:
  if ( v11 )
  {
    v16 = v11[2];
    v17 = *(__int64 **)(v9 + 16);
    v45 = 0;
    if ( (*(_BYTE *)(v16 + 96) & 2) != 0 && v11 != v17 )
    {
      v46[2] = v9;
      v9 = *(_QWORD *)(v16 + 112);
      v46[1] = v11;
      v11 = *(__int64 **)(v9 + 8);
      v17 = *(__int64 **)(v9 + 16);
      v45 = 1;
      v46[0] = v8;
      v8 = v46;
      v16 = v11[2];
    }
    v41 = a1;
    v18 = 0;
    v19 = v12;
    v40 = a2;
    v20 = v14;
    v35 = 0;
    v43 = 0;
    v38 = v15;
    if ( v11 == v17 )
      goto LABEL_21;
LABEL_10:
    v21 = v11[2];
    v22 = *v11;
    v23 = *(_BYTE *)(*(_QWORD *)(v21 + 112) + 25LL);
    if ( !v23 )
    {
LABEL_11:
      if ( v22 || !v8 )
      {
        v24 = v9;
        v25 = v8;
      }
      else
      {
        v24 = v8[2];
        v25 = (_QWORD *)*v8;
        v22 = *(_QWORD *)v8[1];
      }
      v26 = *(_QWORD *)(v16 + 40);
      if ( *(_BYTE *)(v19 + 80) == 16
        && (*(_BYTE *)(v19 + 96) & 8) != 0
        && (v30 = *(__int64 **)(*(_QWORD *)v26 + 96LL), v30[34])
        && (v31 = *v30) != 0 )
      {
        while ( *(_BYTE *)(v31 + 80) != 16 || (*(_BYTE *)(v31 + 96) & 4) == 0 || v41 != **(_QWORD **)(v31 + 88) )
        {
          v31 = *(_QWORD *)(v31 + 16);
          if ( !v31 )
            goto LABEL_15;
        }
        v41 = v31;
        v27 = sub_883C30(v31, v26, v22, v24, v19, v25);
      }
      else
      {
LABEL_15:
        v27 = sub_883C30(v41, v26, v22, v24, v19, v25);
      }
      if ( v27 )
        v18 = 1;
      goto LABEL_18;
    }
    while ( 1 )
    {
      if ( !v20 )
      {
        v36 = v22;
        v37 = v23;
        v29 = sub_87D890(v40);
        v22 = v36;
        v43 = v29;
        v23 = v37;
      }
      v20 = 1;
      if ( v43 )
        goto LABEL_11;
      if ( v23 == 1 )
      {
        if ( !v38 )
        {
          v39 = v22;
          v34 = sub_87D970(v40);
          v22 = v39;
          v35 = v34;
        }
        if ( v35 )
        {
          v38 = 1;
          v20 = 1;
          goto LABEL_11;
        }
        v43 = 0;
        v20 = 1;
        v38 = 1;
      }
LABEL_18:
      if ( !v45 )
        return v18;
      v9 = *(_QWORD *)v9;
      if ( !v9 )
        return v18;
      v11 = *(__int64 **)(v9 + 8);
      v16 = v11[2];
      if ( v11 != *(__int64 **)(v9 + 16) )
        goto LABEL_10;
LABEL_21:
      v23 = *(_BYTE *)(v9 + 25);
      v22 = 0;
      if ( !v23 )
        goto LABEL_11;
    }
  }
  v18 = 0;
  if ( *(_BYTE *)(a1 + 80) != 16 )
    return v18;
  v32 = *(__int64 **)(a1 + 88);
  v33 = *v32;
  if ( *(_BYTE *)(*v32 + 80) == 24 )
    v33 = *(_QWORD *)(v33 + 88);
  return sub_883A10(v33, a1, 0);
}
