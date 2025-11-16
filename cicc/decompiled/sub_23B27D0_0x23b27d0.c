// Function: sub_23B27D0
// Address: 0x23b27d0
//
__int64 __fastcall sub_23B27D0(__int64 *a1)
{
  _QWORD *v2; // rdi
  __int64 v3; // rax
  void *(*v4)(); // rdx
  __int64 v5; // r13
  void *v7; // rax
  _QWORD *v8; // rdi
  __int64 v9; // rax
  void *(*v10)(); // rdx
  void *v11; // rax
  _QWORD v12[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_23B2720(v12, a1);
  v2 = (_QWORD *)v12[0];
  if ( !v12[0] )
    goto LABEL_10;
  v3 = *(_QWORD *)v12[0];
  v4 = *(void *(**)())(*(_QWORD *)v12[0] + 24LL);
  if ( v4 == sub_23AE340 )
  {
    if ( &unk_4CDFBF8 == &unk_4C5D162 )
      goto LABEL_4;
LABEL_9:
    (*(void (**)(void))(v3 + 8))();
    goto LABEL_10;
  }
  v7 = v4();
  v2 = (_QWORD *)v12[0];
  if ( v7 != &unk_4C5D162 )
  {
    if ( !v12[0] )
      goto LABEL_10;
    v3 = *(_QWORD *)v12[0];
    goto LABEL_9;
  }
LABEL_4:
  v5 = v2[1];
  (*(void (__fastcall **)(_QWORD *))(*v2 + 8LL))(v2);
  if ( v5 )
    return v5;
LABEL_10:
  sub_23B2720(v12, a1);
  v8 = (_QWORD *)v12[0];
  if ( !v12[0] )
    return 0;
  v9 = *(_QWORD *)v12[0];
  v10 = *(void *(**)())(*(_QWORD *)v12[0] + 24LL);
  if ( v10 == sub_23AE340 )
  {
    if ( &unk_4CDFBF8 == &unk_4C5D118 )
      goto LABEL_13;
LABEL_19:
    (*(void (**)(void))(v9 + 8))();
    return 0;
  }
  v11 = v10();
  v8 = (_QWORD *)v12[0];
  if ( v11 != &unk_4C5D118 )
  {
    if ( v12[0] )
    {
      v9 = *(_QWORD *)v12[0];
      goto LABEL_19;
    }
    return 0;
  }
LABEL_13:
  v5 = v8[1];
  (*(void (__fastcall **)(_QWORD *))(*v8 + 8LL))(v8);
  if ( !v5 )
    return v5;
  return *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(v5 + 8) + 8LL) + 40LL);
}
