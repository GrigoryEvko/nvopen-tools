// Function: sub_283DC30
// Address: 0x283dc30
//
__int64 __fastcall sub_283DC30(__int64 *a1, __int64 a2, __int64 a3)
{
  bool v5; // zf
  __int64 v6; // rax
  _QWORD *v7; // r12
  unsigned int v8; // r15d
  _QWORD *v9; // rax
  __int64 v10; // rdi
  _QWORD *v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rsi
  _QWORD *v14; // r12
  _QWORD *v15; // rax
  __int64 v16; // rdi
  _QWORD *v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rsi
  _QWORD *v20; // r12
  _QWORD *i; // r15
  _QWORD *v22; // rax
  __int64 v23; // rdi
  _QWORD *v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v29; // [rsp+8h] [rbp-48h]
  _QWORD *j; // [rsp+8h] [rbp-48h]
  _QWORD v31[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( !*a1 )
    return 1;
  v5 = (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a2 + 40LL))(a2) == 0;
  v6 = *a1;
  if ( !v5 )
    goto LABEL_26;
  v7 = *(_QWORD **)v6;
  v29 = *(_QWORD *)v6 + 32LL * *(unsigned int *)(v6 + 8);
  if ( v29 == *(_QWORD *)v6 )
    goto LABEL_26;
  v8 = 1;
  do
  {
    v31[0] = 0;
    v9 = (_QWORD *)sub_22077B0(0x10u);
    if ( v9 )
    {
      v9[1] = a3;
      *v9 = &unk_4A09EA8;
    }
    v10 = v31[0];
    v31[0] = v9;
    if ( v10 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v10 + 8LL))(v10);
    v11 = v7;
    v13 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 32LL))(a2);
    if ( (v7[3] & 2) == 0 )
      v11 = (_QWORD *)*v7;
    v8 &= (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, _QWORD *))(v7[3] & 0xFFFFFFFFFFFFFFF8LL))(
            v11,
            v13,
            v12,
            v31);
    if ( v31[0] )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v31[0] + 8LL))(v31[0]);
    v7 += 4;
  }
  while ( (_QWORD *)v29 != v7 );
  v6 = *a1;
  if ( (_BYTE)v8 )
  {
LABEL_26:
    v20 = *(_QWORD **)(v6 + 288);
    for ( i = &v20[4 * *(unsigned int *)(v6 + 296)]; i != v20; v20 += 4 )
    {
      v31[0] = 0;
      v22 = (_QWORD *)sub_22077B0(0x10u);
      if ( v22 )
      {
        v22[1] = a3;
        *v22 = &unk_4A09EA8;
      }
      v23 = v31[0];
      v31[0] = v22;
      if ( v23 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v23 + 8LL))(v23);
      v24 = v20;
      v26 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 32LL))(a2);
      if ( (v20[3] & 2) == 0 )
        v24 = (_QWORD *)*v20;
      (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD *))(v20[3] & 0xFFFFFFFFFFFFFFF8LL))(v24, v26, v25, v31);
      if ( v31[0] )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v31[0] + 8LL))(v31[0]);
    }
    return 1;
  }
  v14 = *(_QWORD **)(v6 + 144);
  for ( j = &v14[4 * *(unsigned int *)(v6 + 152)]; j != v14; v14 += 4 )
  {
    v31[0] = 0;
    v15 = (_QWORD *)sub_22077B0(0x10u);
    if ( v15 )
    {
      v15[1] = a3;
      *v15 = &unk_4A09EA8;
    }
    v16 = v31[0];
    v31[0] = v15;
    if ( v16 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
    v17 = v14;
    v19 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 32LL))(a2);
    if ( (v14[3] & 2) == 0 )
      v17 = (_QWORD *)*v14;
    (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD *))(v14[3] & 0xFFFFFFFFFFFFFFF8LL))(v17, v19, v18, v31);
    if ( v31[0] )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v31[0] + 8LL))(v31[0]);
  }
  return v8;
}
