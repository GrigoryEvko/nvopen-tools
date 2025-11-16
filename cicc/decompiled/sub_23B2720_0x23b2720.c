// Function: sub_23B2720
// Address: 0x23b2720
//
void __fastcall sub_23B2720(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // r12
  _QWORD *(__fastcall *v3)(_QWORD *, __int64); // rax
  _QWORD *v4; // rax
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD *v8; // rbx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9

  v2 = *a2;
  if ( *a2 )
  {
    v3 = *(_QWORD *(__fastcall **)(_QWORD *, __int64))(*(_QWORD *)v2 + 16LL);
    if ( v3 == sub_23AEE80 )
    {
      v4 = (_QWORD *)sub_22077B0(0x68u);
      v8 = v4;
      if ( v4 )
      {
        *v4 = &unk_4A16218;
        sub_C8CD80((__int64)(v4 + 1), (__int64)(v4 + 5), v2 + 8, v5, v6, v7);
        sub_C8CD80((__int64)(v8 + 7), (__int64)(v8 + 11), v2 + 56, v9, v10, v11);
      }
      *a1 = v8;
    }
    else
    {
      v3(a1, *a2);
    }
  }
  else
  {
    *a1 = 0;
  }
}
