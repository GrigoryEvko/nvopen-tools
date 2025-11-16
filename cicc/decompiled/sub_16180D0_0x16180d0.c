// Function: sub_16180D0
// Address: 0x16180d0
//
void __fastcall sub_16180D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  bool v4; // zf
  __int64 v5; // rax
  __int64 v6; // r8
  _QWORD *v7; // rbx
  __int64 v8; // rax
  _QWORD *v9; // r14
  __int64 v10; // r15
  __int64 v11; // rsi
  __int64 v12; // rax
  _QWORD *v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // rdx
  unsigned int v18; // eax
  __int64 v19; // [rsp+0h] [rbp-40h]
  __int64 v20; // [rsp+0h] [rbp-40h]
  void (__fastcall *v21)(_QWORD *, __int64, _QWORD); // [rsp+8h] [rbp-38h]
  __int64 v22; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 8);
  if ( v3 != *(_QWORD *)a2 )
  {
    while ( (*(int (__fastcall **)(_QWORD))(**(_QWORD **)(v3 - 8) + 40LL))(*(_QWORD *)(v3 - 8)) > 3 )
    {
      sub_160FB80(a2);
      v3 = *(_QWORD *)(a2 + 8);
      if ( *(_QWORD *)a2 == v3 )
        goto LABEL_6;
    }
    v3 = *(_QWORD *)(a2 + 8);
  }
LABEL_6:
  v4 = (*(unsigned int (__fastcall **)(_QWORD))(**(_QWORD **)(v3 - 8) + 40LL))(*(_QWORD *)(v3 - 8)) == 3;
  v5 = *(_QWORD *)(a2 + 8);
  if ( v4 )
  {
    v6 = *(_QWORD *)(v5 - 8);
    if ( !v6 )
      v6 = 160;
    goto LABEL_9;
  }
  v7 = *(_QWORD **)(v5 - 8);
  v8 = sub_22077B0(568);
  v9 = (_QWORD *)v8;
  if ( !v8 )
  {
    v12 = *(_QWORD *)(a2 + 8);
    v11 = *(_QWORD *)a2;
    v10 = 160;
    if ( v12 == *(_QWORD *)a2 )
    {
      v15 = v7[2];
      v16 = 160;
      v10 = 0;
      goto LABEL_15;
    }
    goto LABEL_12;
  }
  *(_QWORD *)(v8 + 8) = 0;
  v10 = v8 + 160;
  *(_DWORD *)(v8 + 24) = 5;
  *(_QWORD *)(v8 + 16) = &unk_4F9E389;
  *(_QWORD *)(v8 + 80) = v8 + 64;
  *(_QWORD *)(v8 + 88) = v8 + 64;
  *(_QWORD *)(v8 + 128) = v8 + 112;
  *(_QWORD *)(v8 + 136) = v8 + 112;
  *(_QWORD *)(v8 + 32) = 0;
  *(_QWORD *)(v8 + 40) = 0;
  *(_QWORD *)v8 = &unk_49EDF20;
  *(_QWORD *)(v8 + 48) = 0;
  *(_DWORD *)(v8 + 64) = 0;
  *(_QWORD *)(v8 + 72) = 0;
  *(_QWORD *)(v8 + 96) = 0;
  *(_DWORD *)(v8 + 112) = 0;
  *(_QWORD *)(v8 + 120) = 0;
  *(_QWORD *)(v8 + 144) = 0;
  *(_BYTE *)(v8 + 152) = 0;
  sub_160E650(v8 + 160);
  v11 = *(_QWORD *)a2;
  v9[20] = &unk_49EDCC8;
  v12 = *(_QWORD *)(a2 + 8);
  *v9 = &unk_49EDC10;
  if ( v11 != v12 )
  {
LABEL_12:
    v13 = v9 + 41;
    do
    {
      v14 = *(_QWORD *)(v12 - 8);
      v12 -= 8;
      *v13++ = v14 + 224;
    }
    while ( v12 != v11 );
  }
  v15 = v7[2];
  v16 = v10;
LABEL_15:
  v17 = *(unsigned int *)(v15 + 120);
  if ( (unsigned int)v17 >= *(_DWORD *)(v15 + 124) )
  {
    v20 = v16;
    v22 = v15;
    sub_16CD150(v15 + 112, v15 + 128, 0, 8);
    v15 = v22;
    v16 = v20;
    v17 = *(unsigned int *)(v22 + 120);
  }
  v19 = v16;
  *(_QWORD *)(*(_QWORD *)(v15 + 112) + 8 * v17) = v10;
  ++*(_DWORD *)(v15 + 120);
  v21 = *(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*v9 + 64LL);
  v18 = (*(__int64 (__fastcall **)(_QWORD *))(*v7 + 40LL))(v7);
  v21(v9, a2, v18);
  sub_16110B0((char **)a2, v10);
  v6 = v19;
LABEL_9:
  sub_1617B20(v6, a1, 1);
}
