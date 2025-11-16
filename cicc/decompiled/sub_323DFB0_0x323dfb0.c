// Function: sub_323DFB0
// Address: 0x323dfb0
//
void __fastcall sub_323DFB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 *v9; // r12
  __int64 *v10; // r13
  __int64 v11; // r15
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r14
  char v16; // r9
  unsigned __int16 v17; // ax
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 *v20; // rdi
  __int64 v21; // rax
  void (*v22)(); // rax
  __int64 v23; // r12
  __int64 v24; // [rsp+8h] [rbp-68h]
  const char *v25; // [rsp+10h] [rbp-60h] BYREF
  char v26; // [rsp+30h] [rbp-40h]
  char v27; // [rsp+31h] [rbp-3Fh]

  if ( *(_DWORD *)(a2 + 248) )
  {
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 176LL))(
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
      a3,
      0);
    if ( (unsigned __int16)sub_3220AA0(a1) > 4u )
    {
      v18 = *(_QWORD *)(a1 + 8);
      v19 = sub_E75E60(*(_QWORD **)(v18 + 224), a3, v6, v7, v8);
      v20 = *(__int64 **)(v18 + 224);
      v11 = v19;
      v21 = *v20;
      v27 = 1;
      v25 = "Offset entry count";
      v22 = *(void (**)())(v21 + 120);
      v26 = 3;
      if ( v22 != nullsub_98 )
        ((void (__fastcall *)(__int64 *, const char **, __int64))v22)(v20, &v25, 1);
      sub_31DCA10(v18, *(_DWORD *)(a2 + 248));
      (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(v18 + 224) + 208LL))(
        *(_QWORD *)(v18 + 224),
        *(_QWORD *)(a2 + 328),
        0);
      v23 = *(_QWORD *)(a2 + 240);
      v24 = v23 + ((unsigned __int64)*(unsigned int *)(a2 + 248) << 6);
      if ( v23 == v24 )
        goto LABEL_14;
      do
      {
        v23 += 64;
        sub_31DF6B0(v18);
        sub_31DCA50(v18);
      }
      while ( v24 != v23 );
      v9 = *(__int64 **)(a2 + 240);
      v10 = &v9[8 * (unsigned __int64)*(unsigned int *)(a2 + 248)];
      if ( v9 == v10 )
        goto LABEL_14;
    }
    else
    {
      v9 = *(__int64 **)(a2 + 240);
      v10 = &v9[8 * (unsigned __int64)*(unsigned int *)(a2 + 248)];
      if ( v9 == v10 )
        return;
      v11 = 0;
    }
    do
    {
      v14 = v9[1];
      v15 = *(_QWORD *)(a1 + 8);
      v16 = 1;
      if ( !*(_BYTE *)(*(_QWORD *)(v14 + 80) + 43LL) )
      {
        v17 = sub_3220AA0(a1);
        v14 = v9[1];
        v16 = v17 > 4u;
      }
      v12 = *v9;
      v13 = (__int64)(v9 + 2);
      v9 += 8;
      sub_323D190(a1, v15, v12, v13, v14, v16);
    }
    while ( v10 != v9 );
LABEL_14:
    if ( v11 )
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 208LL))(
        *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
        v11,
        0);
  }
}
