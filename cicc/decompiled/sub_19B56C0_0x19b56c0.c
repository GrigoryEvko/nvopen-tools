// Function: sub_19B56C0
// Address: 0x19b56c0
//
__int64 __fastcall sub_19B56C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r14
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  int v13; // eax
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  int v23; // eax
  __int64 *v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 *v29; // rdx
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  int v34; // ebx
  __int64 *v35; // r12
  __int64 v36; // [rsp+0h] [rbp-80h]
  __int64 v37; // [rsp+8h] [rbp-78h]
  __int64 v38; // [rsp+10h] [rbp-70h]
  int v39; // [rsp+20h] [rbp-60h]
  int v40; // [rsp+28h] [rbp-58h]
  __int64 *v41[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 *v42; // [rsp+40h] [rbp-40h]

  LODWORD(v3) = 0;
  if ( !(unsigned __int8)sub_1404700(a1, a2) )
  {
    v6 = *(__int64 **)(a1 + 8);
    v7 = *v6;
    v8 = v6[1];
    if ( v7 == v8 )
LABEL_42:
      BUG();
    while ( *(_UNKNOWN **)v7 != &unk_4F9E06C )
    {
      v7 += 16;
      if ( v8 == v7 )
        goto LABEL_42;
    }
    v3 = *(__int64 **)(**(_QWORD **)(a2 + 32) + 56LL);
    v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_4F9E06C);
    v10 = *(__int64 **)(a1 + 8);
    v36 = v9 + 160;
    v11 = *v10;
    v12 = v10[1];
    if ( v11 == v12 )
LABEL_47:
      BUG();
    while ( *(_UNKNOWN **)v11 != &unk_4F9920C )
    {
      v11 += 16;
      if ( v12 == v11 )
        goto LABEL_47;
    }
    v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(
            *(_QWORD *)(v11 + 8),
            &unk_4F9920C);
    v14 = *(__int64 **)(a1 + 8);
    v40 = v13 + 160;
    v15 = *v14;
    v16 = v14[1];
    if ( v15 == v16 )
LABEL_46:
      BUG();
    while ( *(_UNKNOWN **)v15 != &unk_4F9A488 )
    {
      v15 += 16;
      if ( v16 == v15 )
        goto LABEL_46;
    }
    v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(
            *(_QWORD *)(v15 + 8),
            &unk_4F9A488);
    v18 = *(__int64 **)(a1 + 8);
    v19 = *(_QWORD *)(v17 + 160);
    v20 = *v18;
    v21 = v18[1];
    if ( v20 == v21 )
LABEL_44:
      BUG();
    while ( *(_UNKNOWN **)v20 != &unk_4F9D3C0 )
    {
      v20 += 16;
      if ( v21 == v20 )
        goto LABEL_44;
    }
    v22 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v20 + 8) + 104LL))(
            *(_QWORD *)(v20 + 8),
            &unk_4F9D3C0);
    v23 = sub_14A4050(v22, (__int64)v3);
    v24 = *(__int64 **)(a1 + 8);
    v39 = v23;
    v25 = *v24;
    v26 = v24[1];
    if ( v25 == v26 )
LABEL_45:
      BUG();
    while ( *(_UNKNOWN **)v25 != &unk_4F9D764 )
    {
      v25 += 16;
      if ( v26 == v25 )
        goto LABEL_45;
    }
    v27 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v25 + 8) + 104LL))(
            *(_QWORD *)(v25 + 8),
            &unk_4F9D764);
    v28 = sub_14CF090(v27, (__int64)v3);
    v29 = *(__int64 **)(a1 + 8);
    v30 = v28;
    v31 = *v29;
    v32 = v29[1];
    if ( v31 == v32 )
LABEL_43:
      BUG();
    while ( *(_UNKNOWN **)v31 != &unk_4F98D2D )
    {
      v31 += 16;
      if ( v32 == v31 )
        goto LABEL_43;
    }
    v37 = v30;
    v33 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v31 + 8) + 104LL))(
            *(_QWORD *)(v31 + 8),
            &unk_4F98D2D);
    v38 = sub_13A6090(v33);
    sub_143A950(v41, v3);
    v34 = sub_19B4C50((_QWORD *)a2, v36, v40, v19, v39, v37, v38, (__int64)v41, *(_DWORD *)(a1 + 156));
    if ( v34 == 2 )
      sub_1407870(a3, a2);
    v35 = v42;
    LOBYTE(v3) = v34 != 0;
    if ( v42 )
    {
      sub_1368A00(v42);
      j_j___libc_free_0(v35, 8);
    }
  }
  return (unsigned int)v3;
}
