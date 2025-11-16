// Function: sub_1E76B50
// Address: 0x1e76b50
//
__int64 __fastcall sub_1E76B50(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v4; // rsi
  _QWORD *v5; // rax
  _DWORD *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rax
  _DWORD *v10; // r8
  _DWORD *v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 *v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 *v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rsi
  __int64 *v34; // r12
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 (*v38)(); // rax

  v4 = sub_16D5D50();
  v5 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    goto LABEL_69;
  v6 = dword_4FA0208;
  do
  {
    while ( 1 )
    {
      v7 = v5[2];
      v8 = v5[3];
      if ( v4 <= v5[4] )
        break;
      v5 = (_QWORD *)v5[3];
      if ( !v8 )
        goto LABEL_6;
    }
    v6 = v5;
    v5 = (_QWORD *)v5[2];
  }
  while ( v7 );
LABEL_6:
  if ( v6 == dword_4FA0208 )
    goto LABEL_69;
  if ( v4 < *((_QWORD *)v6 + 4) )
    goto LABEL_69;
  v9 = *((_QWORD *)v6 + 7);
  v10 = v6 + 12;
  if ( !v9 )
    goto LABEL_69;
  v4 = (unsigned int)dword_4FC7488;
  v11 = v6 + 12;
  do
  {
    while ( 1 )
    {
      v12 = *(_QWORD *)(v9 + 16);
      v13 = *(_QWORD *)(v9 + 24);
      if ( *(_DWORD *)(v9 + 32) >= dword_4FC7488 )
        break;
      v9 = *(_QWORD *)(v9 + 24);
      if ( !v13 )
        goto LABEL_13;
    }
    v11 = (_DWORD *)v9;
    v9 = *(_QWORD *)(v9 + 16);
  }
  while ( v12 );
LABEL_13:
  if ( v10 != v11 && dword_4FC7488 >= v11[8] && v11[9] )
  {
    if ( byte_4FC7520 )
      goto LABEL_17;
  }
  else
  {
LABEL_69:
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, unsigned __int64))(**(_QWORD **)(a2 + 16) + 144LL))(
           *(_QWORD *)(a2 + 16),
           v4) )
    {
LABEL_17:
      v14 = (__int64 *)a1[9];
      a1[1] = a2;
      v15 = *v14;
      v16 = v14[1];
      if ( v15 == v16 )
LABEL_62:
        BUG();
      while ( *(_UNKNOWN **)v15 != &unk_4FC6A0C )
      {
        v15 += 16;
        if ( v16 == v15 )
          goto LABEL_62;
      }
      v17 = (*(__int64 (__fastcall **)(_QWORD, void *, __int64, __int64))(**(_QWORD **)(v15 + 8) + 104LL))(
              *(_QWORD *)(v15 + 8),
              &unk_4FC6A0C,
              v16,
              v12);
      v18 = (__int64 *)a1[9];
      a1[2] = v17;
      v19 = *v18;
      v20 = v18[1];
      if ( v19 == v20 )
LABEL_63:
        BUG();
      while ( *(_UNKNOWN **)v19 != &unk_4FC62EC )
      {
        v19 += 16;
        if ( v20 == v19 )
          goto LABEL_63;
      }
      v21 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v19 + 8) + 104LL))(
              *(_QWORD *)(v19 + 8),
              &unk_4FC62EC);
      v22 = (__int64 *)a1[9];
      a1[3] = v21;
      v23 = *v22;
      v24 = v22[1];
      if ( v23 == v24 )
LABEL_64:
        BUG();
      while ( *(_UNKNOWN **)v23 != &unk_4FCBA30 )
      {
        v23 += 16;
        if ( v24 == v23 )
          goto LABEL_64;
      }
      v25 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v23 + 8) + 104LL))(
              *(_QWORD *)(v23 + 8),
              &unk_4FCBA30);
      v26 = (__int64 *)a1[9];
      a1[4] = v25;
      v27 = *v26;
      v28 = v26[1];
      if ( v27 == v28 )
LABEL_65:
        BUG();
      while ( *(_UNKNOWN **)v27 != &unk_4F96DB4 )
      {
        v27 += 16;
        if ( v28 == v27 )
          goto LABEL_65;
      }
      v29 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v27 + 8) + 104LL))(
              *(_QWORD *)(v27 + 8),
              &unk_4F96DB4);
      v30 = (__int64 *)a1[9];
      a1[5] = *(_QWORD *)(v29 + 160);
      v31 = *v30;
      v32 = v30[1];
      if ( v31 == v32 )
LABEL_66:
        BUG();
      while ( *(_UNKNOWN **)v31 != &unk_4FC450C )
      {
        v31 += 16;
        if ( v32 == v31 )
          goto LABEL_66;
      }
      a1[6] = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v31 + 8) + 104LL))(
                *(_QWORD *)(v31 + 8),
                &unk_4FC450C);
      if ( byte_4FC7920 )
        sub_1E926D0(a1[1], a1 + 8, "Before machine scheduling.", 1);
      v33 = a1[1];
      sub_1ED7320(a1[7]);
      if ( (__int64 (*)())qword_4FC7640 == sub_1E6BB30 )
      {
        v37 = a1[4];
        v38 = *(__int64 (**)())(*(_QWORD *)v37 + 256LL);
        if ( v38 != sub_1E6BB00 )
        {
          v36 = ((__int64 (__fastcall *)(__int64, _QWORD *))v38)(v37, a1);
          v34 = (__int64 *)v36;
          if ( v36 )
          {
            sub_1E6CAB0((__int64)a1, v36, 0);
            if ( !byte_4FC7920 )
              goto LABEL_43;
LABEL_49:
            sub_1E926D0(a1[1], a1 + 8, "After machine scheduling.", 1);
LABEL_42:
            if ( !v34 )
              return 1;
LABEL_43:
            (*(void (__fastcall **)(__int64 *))(*v34 + 8))(v34);
            return 1;
          }
        }
        v34 = sub_1E76650(a1);
      }
      else
      {
        v34 = (__int64 *)((__int64 (__fastcall *)(_QWORD *, __int64))qword_4FC7640)(a1, v33);
      }
      sub_1E6CAB0((__int64)a1, (__int64)v34, 0);
      if ( !byte_4FC7920 )
        goto LABEL_42;
      goto LABEL_49;
    }
  }
  return 0;
}
