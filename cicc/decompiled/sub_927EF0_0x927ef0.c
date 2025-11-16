// Function: sub_927EF0
// Address: 0x927ef0
//
__int64 __fastcall sub_927EF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r11
  __int64 v7; // rdi
  __int64 (__fastcall *v8)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned int *v15; // rbx
  unsigned int *v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v20; // rdx
  unsigned int v21; // r12d
  __int64 v22; // rax
  __int64 v23; // r12
  unsigned int *v24; // r12
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v29; // [rsp+8h] [rbp-98h]
  unsigned int *v30; // [rsp+8h] [rbp-98h]
  __int64 v31; // [rsp+8h] [rbp-98h]
  _BYTE v32[32]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v33; // [rsp+30h] [rbp-70h]
  _BYTE v34[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v35; // [rsp+60h] [rbp-40h]

  v4 = a1 + 48;
  v5 = *(_QWORD *)(a1 + 32);
  v33 = 257;
  v6 = *(_QWORD *)(v5 + 696);
  if ( v6 == *(_QWORD *)(a2 + 8) )
  {
    v10 = a2;
    goto LABEL_8;
  }
  v7 = *(_QWORD *)(a1 + 128);
  v8 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v7 + 120LL);
  if ( v8 != sub_920130 )
  {
    v31 = v6;
    v27 = v8(v7, 49u, (_BYTE *)a2, v6);
    v6 = v31;
    v10 = v27;
    goto LABEL_7;
  }
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    v29 = v6;
    if ( (unsigned __int8)sub_AC4810(49) )
      v9 = sub_ADAB70(49, a2, v29, 0);
    else
      v9 = sub_AA93C0(49, a2, v29);
    v6 = v29;
    v10 = v9;
LABEL_7:
    if ( v10 )
      goto LABEL_8;
  }
  v35 = 257;
  v10 = sub_B51D30(49, a2, v6, v34, 0, 0);
  if ( (unsigned __int8)sub_920620(v10) )
  {
    v20 = *(_QWORD *)(a1 + 144);
    v21 = *(_DWORD *)(a1 + 152);
    if ( v20 )
      sub_B99FD0(v10, 3, v20);
    sub_B45150(v10, v21);
  }
  (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
    *(_QWORD *)(a1 + 136),
    v10,
    v32,
    *(_QWORD *)(v4 + 56),
    *(_QWORD *)(v4 + 64));
  v22 = *(_QWORD *)(a1 + 48);
  v23 = 16LL * *(unsigned int *)(a1 + 56);
  v30 = (unsigned int *)(v22 + v23);
  if ( v22 != v22 + v23 )
  {
    v24 = *(unsigned int **)(a1 + 48);
    do
    {
      v25 = *((_QWORD *)v24 + 1);
      v26 = *v24;
      v24 += 4;
      sub_B99FD0(v10, v26, v25);
    }
    while ( v30 != v24 );
  }
LABEL_8:
  v33 = 257;
  v35 = 257;
  v11 = sub_BD2C40(72, unk_3F10A14);
  v12 = v11;
  if ( v11 )
  {
    sub_B44260(v11, a3, 60, 1, 0, 0);
    if ( *(_QWORD *)(v12 - 32) )
    {
      v13 = *(_QWORD *)(v12 - 24);
      **(_QWORD **)(v12 - 16) = v13;
      if ( v13 )
        *(_QWORD *)(v13 + 16) = *(_QWORD *)(v12 - 16);
    }
    *(_QWORD *)(v12 - 32) = v10;
    if ( v10 )
    {
      v14 = *(_QWORD *)(v10 + 16);
      *(_QWORD *)(v12 - 24) = v14;
      if ( v14 )
        *(_QWORD *)(v14 + 16) = v12 - 24;
      *(_QWORD *)(v12 - 16) = v10 + 16;
      *(_QWORD *)(v10 + 16) = v12 - 32;
    }
    sub_BD6B50(v12, v34);
  }
  (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
    *(_QWORD *)(a1 + 136),
    v12,
    v32,
    *(_QWORD *)(v4 + 56),
    *(_QWORD *)(v4 + 64));
  v15 = *(unsigned int **)(a1 + 48);
  v16 = &v15[4 * *(unsigned int *)(a1 + 56)];
  while ( v16 != v15 )
  {
    v17 = *((_QWORD *)v15 + 1);
    v18 = *v15;
    v15 += 4;
    sub_B99FD0(v12, v18, v17);
  }
  return v12;
}
