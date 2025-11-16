// Function: sub_31EB240
// Address: 0x31eb240
//
void __fastcall sub_31EB240(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // r13
  __int64 *v6; // rbx
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rax
  void (__fastcall *v14)(__int64, __int64, _QWORD); // [rsp+8h] [rbp-38h]

  v5 = (__int64 *)(a2 + 8 * a3);
  v6 = (__int64 *)a2;
  v7 = **(_QWORD **)(a1 + 232);
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 208) + 18LL) )
  {
    v10 = *(_QWORD *)(a1 + 224);
    v14 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v10 + 208LL);
    v11 = sub_E6C350(*(_QWORD *)(a1 + 216), a2, a3, (__int64)v14, a5);
    v14(v10, v11, 0);
    if ( v5 != (__int64 *)a2 )
    {
      do
      {
        v12 = *v6++;
        v13 = sub_B2BEC0(v7);
        sub_31EA6F0(a1, v13, v12, 0);
      }
      while ( v5 != v6 );
    }
    (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(
      *(_QWORD *)(a1 + 224),
      *(_QWORD *)(a1 + 280),
      20);
  }
  else
  {
    while ( v5 != v6 )
    {
      v8 = *v6++;
      v9 = sub_B2BEC0(v7);
      sub_31EA6F0(a1, v9, v8, 0);
    }
  }
}
