// Function: sub_2CF5620
// Address: 0x2cf5620
//
__int64 __fastcall sub_2CF5620(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rcx
  bool v12; // zf
  char v13; // al
  __int16 v15; // [rsp+Eh] [rbp-22h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_16:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F8144C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_16;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F8144C);
  v6 = *(__int64 **)(a1 + 8);
  v7 = v5 + 176;
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_15:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F8662C )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_15;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F8662C);
  v11 = sub_CFFAC0(v10, a2);
  v12 = *(_BYTE *)(a1 + 169) == 0;
  LOBYTE(v15) = *(_BYTE *)(a1 + 170);
  v13 = 1;
  if ( !v12 )
    v13 = byte_50145E8;
  HIBYTE(v15) = v13;
  return sub_2CF5350(&v15, a2, v7, v11);
}
