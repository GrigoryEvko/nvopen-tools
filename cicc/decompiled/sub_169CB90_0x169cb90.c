// Function: sub_169CB90
// Address: 0x169cb90
//
bool __fastcall sub_169CB90(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  void *v4; // r15
  __int64 v6; // r12
  __int64 v7; // r13
  void *v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rsi
  char v11; // al
  __int64 v12; // rcx
  __int64 v13; // rax
  void *v14; // rsi
  void *v15; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(_QWORD *)(a2 + 8);
  v4 = *(void **)(v2 + 8);
  if ( *(void **)(v3 + 8) == v4 )
  {
    v6 = a1;
    v7 = a2;
    v8 = sub_16982C0();
    do
    {
      v15 = v8;
      v9 = v2 + 8;
      v10 = v3 + 8;
      v11 = v8 == v4 ? sub_169CB90(v9, v10) : sub_1698510(v9, v10);
      v8 = v15;
      if ( !v11 )
        break;
      v12 = *(_QWORD *)(v6 + 8);
      v13 = *(_QWORD *)(v7 + 8);
      v14 = *(void **)(v12 + 40);
      if ( v14 != *(void **)(v13 + 40) )
        break;
      v6 = v12 + 40;
      v7 = v13 + 40;
      if ( v15 != v14 )
        return sub_1698510(v12 + 40, v13 + 40);
      v2 = *(_QWORD *)(v12 + 48);
      v3 = *(_QWORD *)(v13 + 48);
      v4 = *(void **)(v2 + 8);
    }
    while ( v4 == *(void **)(v3 + 8) );
  }
  return 0;
}
