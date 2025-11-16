// Function: sub_29DBAC0
// Address: 0x29dbac0
//
bool __fastcall sub_29DBAC0(__int64 a1, __int64 a2, __int64 a3)
{
  bool result; // al
  unsigned __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rbx
  _QWORD *v7; // rdi
  __int64 v8; // rcx
  _QWORD *v9; // rax
  _BYTE *v10; // rax
  __int64 v11; // [rsp-38h] [rbp-38h]
  _QWORD v12[5]; // [rsp-28h] [rbp-28h] BYREF

  if ( *(_BYTE *)a2 == 2 )
    return 0;
  if ( *(_BYTE *)a2 == 1 )
  {
    v11 = a3;
    v10 = (_BYTE *)sub_B325F0(a2);
    a3 = v11;
    if ( *v10 == 2 )
      return 0;
  }
  result = 1;
  if ( *(_QWORD *)(a1 + 16) )
    return result;
  if ( !*(_BYTE *)(a1 + 24) )
    return 0;
  v4 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  v5 = *(_QWORD *)(a2 + 40);
  v6 = *(_QWORD *)(v4 + 32);
  v7 = *(_QWORD **)(v4 + 24);
  v8 = *(_QWORD *)(v5 + 176);
  v12[0] = *(_QWORD *)(v5 + 168);
  v12[1] = v8;
  v9 = sub_29DB840(v7, v6, (__int64)v12);
  if ( (_QWORD *)v6 == v9 )
    BUG();
  return (*(_BYTE *)(*v9 + 12LL) & 0xFu) - 7 > 1;
}
