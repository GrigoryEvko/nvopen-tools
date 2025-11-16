// Function: sub_94C070
// Address: 0x94c070
//
__int64 __fastcall sub_94C070(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, unsigned int a5, unsigned int a6)
{
  unsigned int v8; // r14d
  __int64 v9; // r13
  __int64 v10; // rax
  unsigned __int8 v11; // al
  __int64 v12; // rax
  __int64 v13; // r9
  __int64 v14; // r12
  unsigned int *v15; // rax
  unsigned int *v16; // r13
  unsigned int *v17; // r14
  __int64 v18; // rdx
  __int64 v19; // rsi
  unsigned int v20; // r14d
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rax
  int v24; // r9d
  __int64 v25; // r12
  unsigned int *v26; // rax
  unsigned int *v27; // r13
  unsigned int *v28; // r14
  __int64 v29; // rdx
  __int64 v30; // rsi
  __int64 v31; // r13
  __int64 v32; // rax
  unsigned __int8 v33; // al
  __int64 v34; // rax
  int v35; // r9d
  __int64 v36; // r12
  unsigned int *v37; // rax
  unsigned int *v38; // r13
  unsigned int *v39; // rbx
  __int64 v40; // rdx
  __int64 v41; // rsi
  __int64 v43; // [rsp-10h] [rbp-A0h]
  int v44; // [rsp+8h] [rbp-88h]
  int v45; // [rsp+8h] [rbp-88h]
  unsigned int v46; // [rsp+8h] [rbp-88h]
  int v49; // [rsp+14h] [rbp-7Ch]
  _BYTE v51[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v52; // [rsp+50h] [rbp-40h]

  v8 = (unsigned int)sub_92F410(a2, a3);
  v9 = sub_94BE50(a2, a4);
  v10 = sub_AA4E30(*(_QWORD *)(a2 + 96));
  v11 = sub_AE5020(v10, *(_QWORD *)(v9 + 8));
  v52 = 257;
  v44 = v11;
  v12 = sub_BD2C40(80, unk_3F10A10);
  v14 = v12;
  if ( v12 )
  {
    sub_B4D3C0(v12, v9, v8, 0, v44, v13, 0, 0);
    v13 = v43;
  }
  (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a2 + 136) + 16LL))(
    *(_QWORD *)(a2 + 136),
    v14,
    v51,
    *(_QWORD *)(a2 + 104),
    *(_QWORD *)(a2 + 112),
    v13);
  v15 = *(unsigned int **)(a2 + 48);
  v16 = &v15[4 * *(unsigned int *)(a2 + 56)];
  if ( v15 != v16 )
  {
    v17 = *(unsigned int **)(a2 + 48);
    do
    {
      v18 = *((_QWORD *)v17 + 1);
      v19 = *v17;
      v17 += 4;
      sub_B99FD0(v14, v19, v18);
    }
    while ( v16 != v17 );
  }
  v20 = (unsigned int)sub_92F410(a2, *(_QWORD *)(a3 + 16));
  v21 = sub_94BE50(a2, a5);
  v22 = sub_AA4E30(*(_QWORD *)(a2 + 96));
  v45 = (unsigned __int8)sub_AE5020(v22, *(_QWORD *)(v21 + 8));
  v52 = 257;
  v23 = sub_BD2C40(80, unk_3F10A10);
  v25 = v23;
  if ( v23 )
    sub_B4D3C0(v23, v21, v20, 0, v45, v24, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
    *(_QWORD *)(a2 + 136),
    v25,
    v51,
    *(_QWORD *)(a2 + 104),
    *(_QWORD *)(a2 + 112));
  v26 = *(unsigned int **)(a2 + 48);
  v27 = &v26[4 * *(unsigned int *)(a2 + 56)];
  if ( v26 != v27 )
  {
    v28 = *(unsigned int **)(a2 + 48);
    do
    {
      v29 = *((_QWORD *)v28 + 1);
      v30 = *v28;
      v28 += 4;
      sub_B99FD0(v25, v30, v29);
    }
    while ( v27 != v28 );
  }
  v46 = (unsigned int)sub_92F410(a2, *(_QWORD *)(*(_QWORD *)(a3 + 16) + 16LL));
  v31 = sub_94BE50(a2, a6);
  v32 = sub_AA4E30(*(_QWORD *)(a2 + 96));
  v33 = sub_AE5020(v32, *(_QWORD *)(v31 + 8));
  v52 = 257;
  v49 = v33;
  v34 = sub_BD2C40(80, unk_3F10A10);
  v36 = v34;
  if ( v34 )
    sub_B4D3C0(v34, v31, v46, 0, v49, v35, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
    *(_QWORD *)(a2 + 136),
    v36,
    v51,
    *(_QWORD *)(a2 + 104),
    *(_QWORD *)(a2 + 112));
  v37 = *(unsigned int **)(a2 + 48);
  v38 = &v37[4 * *(unsigned int *)(a2 + 56)];
  if ( v37 != v38 )
  {
    v39 = *(unsigned int **)(a2 + 48);
    do
    {
      v40 = *((_QWORD *)v39 + 1);
      v41 = *v39;
      v39 += 4;
      sub_B99FD0(v36, v41, v40);
    }
    while ( v38 != v39 );
  }
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
