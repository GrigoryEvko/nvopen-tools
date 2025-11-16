// Function: sub_1E74C70
// Address: 0x1e74c70
//
__int64 __fastcall sub_1E74C70(_QWORD *a1, __int64 a2)
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
  __int64 v22; // rdi
  __int64 (*v23)(); // rax
  _QWORD *v24; // r12
  __int64 v26; // rax

  v4 = sub_16D5D50();
  v5 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    goto LABEL_44;
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
    goto LABEL_44;
  if ( v4 < *((_QWORD *)v6 + 4) )
    goto LABEL_44;
  v9 = *((_QWORD *)v6 + 7);
  v10 = v6 + 12;
  if ( !v9 )
    goto LABEL_44;
  v4 = (unsigned int)dword_4FC73A8;
  v11 = v6 + 12;
  do
  {
    while ( 1 )
    {
      v12 = *(_QWORD *)(v9 + 16);
      v13 = *(_QWORD *)(v9 + 24);
      if ( *(_DWORD *)(v9 + 32) >= dword_4FC73A8 )
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
  if ( v10 != v11 && dword_4FC73A8 >= v11[8] && v11[9] )
  {
    if ( byte_4FC7440 )
      goto LABEL_17;
  }
  else
  {
LABEL_44:
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, unsigned __int64))(**(_QWORD **)(a2 + 16) + 176LL))(
           *(_QWORD *)(a2 + 16),
           v4) )
    {
LABEL_17:
      v14 = (__int64 *)a1[9];
      a1[1] = a2;
      v15 = *v14;
      v16 = v14[1];
      if ( v15 == v16 )
LABEL_41:
        BUG();
      while ( *(_UNKNOWN **)v15 != &unk_4FC6A0C )
      {
        v15 += 16;
        if ( v16 == v15 )
          goto LABEL_41;
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
LABEL_40:
        BUG();
      while ( *(_UNKNOWN **)v19 != &unk_4FCBA30 )
      {
        v19 += 16;
        if ( v20 == v19 )
          goto LABEL_40;
      }
      v21 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v19 + 8) + 104LL))(
              *(_QWORD *)(v19 + 8),
              &unk_4FCBA30);
      a1[4] = v21;
      v22 = v21;
      if ( byte_4FC7920 )
      {
        sub_1E926D0(a1[1], a1 + 8, "Before post machine scheduling.", 1);
        v22 = a1[4];
      }
      v23 = *(__int64 (**)())(*(_QWORD *)v22 + 264LL);
      if ( v23 == sub_1E6BB10
        || (v26 = ((__int64 (__fastcall *)(__int64, _QWORD *))v23)(v22, a1), (v24 = (_QWORD *)v26) == 0) )
      {
        v24 = sub_1E74930(a1);
        sub_1E6CAB0((__int64)a1, (__int64)v24, 1);
        if ( !byte_4FC7920 )
          goto LABEL_29;
      }
      else
      {
        sub_1E6CAB0((__int64)a1, v26, 1);
        if ( !byte_4FC7920 )
          goto LABEL_30;
      }
      sub_1E926D0(a1[1], a1 + 8, "After post machine scheduling.", 1);
LABEL_29:
      if ( !v24 )
        return 1;
LABEL_30:
      (*(void (__fastcall **)(_QWORD *))(*v24 + 8LL))(v24);
      return 1;
    }
  }
  return 0;
}
