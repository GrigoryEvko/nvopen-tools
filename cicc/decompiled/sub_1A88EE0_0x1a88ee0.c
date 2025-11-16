// Function: sub_1A88EE0
// Address: 0x1a88ee0
//
__int64 __fastcall sub_1A88EE0(__int64 a1, __int64 a2, __m128i a3, __m128i a4)
{
  unsigned int v4; // eax
  unsigned int v5; // r14d
  __int64 *v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 *v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 *v15; // rcx
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rcx
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 *v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  unsigned __int64 v31; // rsi
  _QWORD *v32; // rax
  _DWORD *v33; // rdi
  __int64 v34; // rcx
  __int64 v35; // rdx
  __int64 v36; // rax
  _DWORD *v37; // r8
  _DWORD *v38; // rdi
  __int64 v39; // rcx
  __int64 v40; // rdx
  int v41; // eax
  __int64 v42; // [rsp+0h] [rbp-70h]
  __int64 v43; // [rsp+8h] [rbp-68h]
  _QWORD v44[3]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v45; // [rsp+28h] [rbp-48h]
  __int64 v46; // [rsp+30h] [rbp-40h]

  v4 = sub_1636880(a1, a2);
  if ( (_BYTE)v4 )
  {
    return 0;
  }
  else
  {
    v7 = *(__int64 **)(a1 + 8);
    v5 = v4;
    v8 = *v7;
    v9 = v7[1];
    if ( v8 == v9 )
LABEL_52:
      BUG();
    while ( *(_UNKNOWN **)v8 != &unk_4F9920C )
    {
      v8 += 16;
      if ( v9 == v8 )
        goto LABEL_52;
    }
    v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F9920C);
    v11 = *(__int64 **)(a1 + 8);
    v42 = v10 + 160;
    v12 = *v11;
    v13 = v11[1];
    if ( v12 == v13 )
LABEL_54:
      BUG();
    while ( *(_UNKNOWN **)v12 != &unk_4F9A488 )
    {
      v12 += 16;
      if ( v13 == v12 )
        goto LABEL_54;
    }
    v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(
            *(_QWORD *)(v12 + 8),
            &unk_4F9A488);
    v15 = *(__int64 **)(a1 + 8);
    v16 = *(_QWORD *)(v14 + 160);
    v17 = *v15;
    v18 = v15[1];
    if ( v17 == v18 )
LABEL_55:
      BUG();
    while ( *(_UNKNOWN **)v17 != &unk_4F9D764 )
    {
      v17 += 16;
      if ( v18 == v17 )
        goto LABEL_55;
    }
    v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(
            *(_QWORD *)(v17 + 8),
            &unk_4F9D764);
    v20 = sub_14CF090(v19, a2);
    v21 = *(__int64 **)(a1 + 8);
    v22 = v20;
    v23 = *v21;
    v24 = v21[1];
    if ( v23 == v24 )
LABEL_56:
      BUG();
    while ( *(_UNKNOWN **)v23 != &unk_4F99CB0 )
    {
      v23 += 16;
      if ( v24 == v23 )
        goto LABEL_56;
    }
    v25 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v23 + 8) + 104LL))(
            *(_QWORD *)(v23 + 8),
            &unk_4F99CB0);
    v26 = *(__int64 **)(a1 + 8);
    v27 = *(_QWORD *)(v25 + 160);
    v28 = *v26;
    v29 = v26[1];
    if ( v28 == v29 )
LABEL_53:
      BUG();
    while ( *(_UNKNOWN **)v28 != &unk_4F9D3C0 )
    {
      v28 += 16;
      if ( v29 == v28 )
        goto LABEL_53;
    }
    v43 = v27;
    v30 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v28 + 8) + 104LL))(
            *(_QWORD *)(v28 + 8),
            &unk_4F9D3C0);
    v44[0] = v22;
    v44[2] = v16;
    v44[1] = v42;
    v45 = sub_14A4050(v30, a2);
    v46 = v43;
    v31 = sub_16D5D50();
    v32 = *(_QWORD **)&dword_4FA0208[2];
    if ( !*(_QWORD *)&dword_4FA0208[2] )
      goto LABEL_42;
    v33 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v34 = v32[2];
        v35 = v32[3];
        if ( v31 <= v32[4] )
          break;
        v32 = (_QWORD *)v32[3];
        if ( !v35 )
          goto LABEL_29;
      }
      v33 = v32;
      v32 = (_QWORD *)v32[2];
    }
    while ( v34 );
LABEL_29:
    if ( v33 == dword_4FA0208 )
      goto LABEL_42;
    if ( v31 < *((_QWORD *)v33 + 4) )
      goto LABEL_42;
    v36 = *((_QWORD *)v33 + 7);
    v37 = v33 + 12;
    if ( !v36 )
      goto LABEL_42;
    v38 = v33 + 12;
    do
    {
      while ( 1 )
      {
        v39 = *(_QWORD *)(v36 + 16);
        v40 = *(_QWORD *)(v36 + 24);
        if ( *(_DWORD *)(v36 + 32) >= dword_4FB5108 )
          break;
        v36 = *(_QWORD *)(v36 + 24);
        if ( !v40 )
          goto LABEL_36;
      }
      v38 = (_DWORD *)v36;
      v36 = *(_QWORD *)(v36 + 16);
    }
    while ( v39 );
LABEL_36:
    if ( v37 == v38 || dword_4FB5108 < v38[8] || (int)v38[9] <= 0 )
LABEL_42:
      v41 = sub_14A3290(v45);
    else
      v41 = dword_4FB51A0;
    if ( v41 )
      return (unsigned int)sub_1A87830((__int64)v44, a3, a4);
  }
  return v5;
}
