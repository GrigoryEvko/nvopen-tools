// Function: sub_C3E590
// Address: 0xc3e590
//
bool __fastcall sub_C3E590(__int64 a1, __int64 a2)
{
  void **v2; // r15
  void **v3; // r14
  void *v4; // rbx
  __int64 v6; // r12
  __int64 v7; // r13
  void *v8; // rdx
  char v9; // al
  __int64 v10; // rcx
  __int64 v11; // rax
  void *v12; // rsi
  void *v13; // [rsp+8h] [rbp-38h]

  v2 = *(void ***)(a1 + 8);
  v3 = *(void ***)(a2 + 8);
  v4 = *v2;
  if ( *v2 == *v3 )
  {
    v6 = a1;
    v7 = a2;
    v8 = sub_C33340();
    do
    {
      v13 = v8;
      v9 = v8 == v4 ? sub_C3E590(v2) : sub_C33D00((__int64)v2, (__int64)v3);
      v8 = v13;
      if ( !v9 )
        break;
      v10 = *(_QWORD *)(v6 + 8);
      v11 = *(_QWORD *)(v7 + 8);
      v12 = *(void **)(v10 + 24);
      if ( v12 != *(void **)(v11 + 24) )
        break;
      v6 = v10 + 24;
      v7 = v11 + 24;
      if ( v13 != v12 )
        return sub_C33D00(v10 + 24, v11 + 24);
      v2 = *(void ***)(v10 + 32);
      v3 = *(void ***)(v11 + 32);
      v4 = *v2;
    }
    while ( *v2 == *v3 );
  }
  return 0;
}
