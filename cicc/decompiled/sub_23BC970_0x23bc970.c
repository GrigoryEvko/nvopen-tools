// Function: sub_23BC970
// Address: 0x23bc970
//
void __fastcall sub_23BC970(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __m128i **a5)
{
  __int64 v6; // r12
  _QWORD *(__fastcall *v7)(__int64 *, __int64); // rax
  _QWORD *v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rbx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16[5]; // [rsp+18h] [rbp-28h] BYREF

  v6 = *a2;
  if ( *a2 )
  {
    v7 = *(_QWORD *(__fastcall **)(__int64 *, __int64))(*(_QWORD *)v6 + 16LL);
    if ( v7 == sub_23AEE80 )
    {
      v8 = (_QWORD *)sub_22077B0(0x68u);
      v12 = (__int64)v8;
      if ( v8 )
      {
        *v8 = &unk_4A16218;
        sub_C8CD80((__int64)(v8 + 1), (__int64)(v8 + 5), v6 + 8, v9, v10, v11);
        sub_C8CD80(v12 + 56, v12 + 88, v6 + 56, v13, v14, v15);
      }
      v16[0] = v12;
    }
    else
    {
      v7(v16, *a2);
    }
  }
  else
  {
    v16[0] = 0;
  }
  sub_23BC790(v16, a5);
  if ( v16[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v16[0] + 8LL))(v16[0]);
}
