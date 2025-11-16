// Function: sub_323CCF0
// Address: 0x323ccf0
//
void __fastcall sub_323CCF0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // rdi
  __int64 *v7; // rdx
  __int64 v8; // r14
  __int64 *v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // r8
  __int64 v13; // rcx
  __int64 *v14; // r13
  char v15; // r9
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rdi
  void (*v21)(); // rax
  __int64 v22; // r14
  __int64 v23; // [rsp-78h] [rbp-78h]
  __int64 *v24; // [rsp-70h] [rbp-70h]
  __int64 v25; // [rsp-70h] [rbp-70h]
  _QWORD v26[4]; // [rsp-68h] [rbp-68h] BYREF
  char v27; // [rsp-48h] [rbp-48h]
  char v28; // [rsp-47h] [rbp-47h]

  if ( *(_DWORD *)(a1 + 1296) )
  {
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 176LL))(
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
      a2,
      0);
    if ( (unsigned __int16)sub_3220AA0(a1) > 4u )
    {
      v18 = *(_QWORD *)(a1 + 8);
      v19 = sub_E75E60(*(_QWORD **)(v18 + 224), a2, v3, v4, v5);
      v20 = *(_QWORD *)(v18 + 224);
      v23 = v19;
      v21 = *(void (**)())(*(_QWORD *)v20 + 120LL);
      v28 = 1;
      v26[0] = "Offset entry count";
      v27 = 3;
      if ( v21 != nullsub_98 )
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v21)(v20, v26, 1);
      sub_31DCA10(v18, *(_DWORD *)(a1 + 1296));
      (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(v18 + 224) + 208LL))(
        *(_QWORD *)(v18 + 224),
        *(_QWORD *)(a1 + 2776),
        0);
      v22 = *(_QWORD *)(a1 + 1288);
      v25 = v22 + 32LL * *(unsigned int *)(a1 + 1296);
      if ( v22 == v25 )
        goto LABEL_16;
      do
      {
        v22 += 32;
        sub_31DF6B0(v18);
        sub_31DCA50(v18);
      }
      while ( v25 != v22 );
      v6 = *(unsigned int *)(a1 + 1296);
      v7 = *(__int64 **)(a1 + 1288);
      v24 = &v7[4 * v6];
      if ( v7 == v24 )
        goto LABEL_16;
    }
    else
    {
      v6 = *(unsigned int *)(a1 + 1296);
      v7 = *(__int64 **)(a1 + 1288);
      v8 = 4LL * *(unsigned int *)(a1 + 1296);
      v24 = &v7[v8];
      if ( &v7[v8] == v7 )
        return;
      v23 = 0;
    }
    v9 = v7;
    while ( 1 )
    {
      v11 = *(_QWORD *)(a1 + 8);
      v12 = *v9;
      v13 = *(_QWORD *)(a1 + 1432);
      v14 = v9;
      v15 = (unsigned int)(*(_DWORD *)(*(_QWORD *)(v11 + 200) + 544LL) - 42) > 1;
      v16 = (char *)v9 - (char *)v7;
      v17 = v9[2];
      v10 = (v16 >> 5) + 1 == v6 ? *(unsigned int *)(a1 + 1440) : v9[6];
      v26[1] = v10 - v17;
      v9 += 4;
      v26[0] = v13 + 32 * v17;
      sub_323BE90(a1, v11, *(v9 - 3), (__int64)v26, v12, v15, a1, v14);
      if ( v24 == v9 )
        break;
      v7 = *(__int64 **)(a1 + 1288);
      v6 = *(unsigned int *)(a1 + 1296);
    }
LABEL_16:
    if ( v23 )
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 208LL))(
        *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
        v23,
        0);
  }
}
