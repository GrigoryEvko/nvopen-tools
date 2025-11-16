// Function: sub_19BFF20
// Address: 0x19bff20
//
__int64 __fastcall sub_19BFF20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 *v9; // r15
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 *v19; // rdx
  __int64 v20; // r10
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 *v26; // r8
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  unsigned __int8 v30; // al
  int v31; // ebx
  __int64 *v32; // r12
  __int64 *v33; // [rsp+8h] [rbp-98h]
  __int64 v34; // [rsp+18h] [rbp-88h]
  __int64 v35; // [rsp+20h] [rbp-80h]
  __int64 v36; // [rsp+28h] [rbp-78h]
  __int8 v37; // [rsp+38h] [rbp-68h] BYREF
  char v39; // [rsp+3Ah] [rbp-66h] BYREF
  char v41; // [rsp+3Ch] [rbp-64h] BYREF
  char v43; // [rsp+3Eh] [rbp-62h] BYREF
  int v45; // [rsp+40h] [rbp-60h] BYREF
  int v47; // [rsp+48h] [rbp-58h] BYREF
  __int64 *v49[2]; // [rsp+50h] [rbp-50h] BYREF
  __int64 *v50; // [rsp+60h] [rbp-40h]

  LODWORD(v3) = 0;
  if ( !(unsigned __int8)sub_1404700(a1, a2) )
  {
    v6 = *(__int64 **)(a1 + 8);
    v7 = *v6;
    v8 = v6[1];
    if ( v7 == v8 )
LABEL_48:
      BUG();
    while ( *(_UNKNOWN **)v7 != &unk_4F9E06C )
    {
      v7 += 16;
      if ( v8 == v7 )
        goto LABEL_48;
    }
    v9 = *(__int64 **)(**(_QWORD **)(a2 + 32) + 56LL);
    v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_4F9E06C);
    v11 = *(__int64 **)(a1 + 8);
    v36 = v10 + 160;
    v12 = *v11;
    v13 = v11[1];
    if ( v12 == v13 )
LABEL_50:
      BUG();
    while ( *(_UNKNOWN **)v12 != &unk_4F9920C )
    {
      v12 += 16;
      if ( v13 == v12 )
        goto LABEL_50;
    }
    v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(
            *(_QWORD *)(v12 + 8),
            &unk_4F9920C);
    v15 = *(__int64 **)(a1 + 8);
    v3 = v14 + 160;
    v16 = *v15;
    v17 = v15[1];
    if ( v16 == v17 )
LABEL_51:
      BUG();
    while ( *(_UNKNOWN **)v16 != &unk_4F9A488 )
    {
      v16 += 16;
      if ( v17 == v16 )
        goto LABEL_51;
    }
    v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(
            *(_QWORD *)(v16 + 8),
            &unk_4F9A488);
    v19 = *(__int64 **)(a1 + 8);
    v20 = *(_QWORD *)(v18 + 160);
    v21 = *v19;
    v22 = v19[1];
    if ( v21 == v22 )
LABEL_52:
      BUG();
    while ( *(_UNKNOWN **)v21 != &unk_4F9D3C0 )
    {
      v21 += 16;
      if ( v22 == v21 )
        goto LABEL_52;
    }
    v35 = v20;
    v23 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v21 + 8) + 104LL))(
            *(_QWORD *)(v21 + 8),
            &unk_4F9D3C0);
    v24 = sub_14A4050(v23, (__int64)v9);
    v25 = *(__int64 **)(a1 + 8);
    v26 = (__int64 *)v24;
    v27 = *v25;
    v28 = v25[1];
    if ( v27 == v28 )
LABEL_49:
      BUG();
    while ( *(_UNKNOWN **)v27 != &unk_4F9D764 )
    {
      v27 += 16;
      if ( v28 == v27 )
        goto LABEL_49;
    }
    v33 = v26;
    v29 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v27 + 8) + 104LL))(
            *(_QWORD *)(v27 + 8),
            &unk_4F9D764);
    v34 = sub_14CF090(v29, (__int64)v9);
    sub_143A950(v49, v9);
    v30 = sub_1636850(a1, (__int64)&unk_4FB65F4);
    if ( *(_BYTE *)(a1 + 183) )
      v37 = *(_BYTE *)(a1 + 182);
    if ( *(_BYTE *)(a1 + 181) )
      v39 = *(_BYTE *)(a1 + 180);
    if ( *(_BYTE *)(a1 + 179) )
      v41 = *(_BYTE *)(a1 + 178);
    if ( *(_BYTE *)(a1 + 177) )
      v43 = *(_BYTE *)(a1 + 176);
    if ( *(_BYTE *)(a1 + 172) )
      v45 = *(_DWORD *)(a1 + 168);
    if ( *(_BYTE *)(a1 + 164) )
      v47 = *(_DWORD *)(a1 + 160);
    v31 = sub_19BE360(
            a2,
            v36,
            v3,
            v35,
            v33,
            v34,
            (__int64 *)v49,
            v30,
            *(_DWORD *)(a1 + 156),
            (__int64)&v47,
            (__int64)&v45,
            &v43,
            &v41,
            &v39,
            &v37);
    if ( v31 == 2 )
      sub_1407870(a3, a2);
    v32 = v50;
    LOBYTE(v3) = v31 != 0;
    if ( v50 )
    {
      sub_1368A00(v50);
      j_j___libc_free_0(v32, 8);
    }
  }
  return (unsigned int)v3;
}
