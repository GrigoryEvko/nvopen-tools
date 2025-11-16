// Function: sub_23BFC70
// Address: 0x23bfc70
//
void __fastcall sub_23BFC70(__int64 *a1, __m128i **a2)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 i; // r12
  __int64 v6; // rsi
  void *(*v7)(); // rax
  void *v8; // rax
  __int64 v9; // r12
  __int64 v10; // rsi
  void *(*v11)(); // rax
  void *v12; // rax
  __int64 v13; // rbx
  void *(*v14)(); // rax
  __int64 v15; // r12
  __int64 v16[7]; // [rsp+8h] [rbp-38h] BYREF

  sub_23B2720(v16, a1);
  v3 = sub_23B27D0(v16);
  if ( v16[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v16[0] + 8LL))(v16[0]);
  if ( !v3 )
  {
    sub_23B2720(v16, a1);
    if ( v16[0]
      && ((v7 = *(void *(**)())(*(_QWORD *)v16[0] + 24LL), v7 != sub_23AE340) ? (v8 = v7()) : (v8 = &unk_4CDFBF8),
          v8 == &unk_4C5D161) )
    {
      v9 = *(_QWORD *)(v16[0] + 8);
      sub_23B42E0(v16);
      v10 = v9;
      if ( v9 )
      {
LABEL_14:
        sub_23BF170(a2, v10);
        return;
      }
    }
    else
    {
      sub_23B42E0(v16);
    }
    sub_23B2720(v16, a1);
    if ( v16[0]
      && ((v11 = *(void *(**)())(*(_QWORD *)v16[0] + 24LL), v11 != sub_23AE340) ? (v12 = v11()) : (v12 = &unk_4CDFBF8),
          v12 == &unk_4C5D160) )
    {
      v13 = *(_QWORD *)(v16[0] + 8);
      sub_23B42E0(v16);
      if ( v13 )
      {
        v10 = *(_QWORD *)(**(_QWORD **)(v13 + 32) + 72LL);
        goto LABEL_14;
      }
    }
    else
    {
      sub_23B42E0(v16);
    }
    sub_23B2720(v16, a1);
    if ( v16[0] && (v14 = *(void *(**)())(*(_QWORD *)v16[0] + 24LL), v14 != sub_23AE340) && v14() == &unk_4CDFC40 )
    {
      v15 = *(_QWORD *)(v16[0] + 8);
      sub_23B42E0(v16);
      if ( v15 )
      {
        sub_23BE720(a2, v15);
        return;
      }
    }
    else
    {
      sub_23B42E0(v16);
    }
    BUG();
  }
  v4 = *(_QWORD *)(v3 + 32);
  for ( i = v3 + 24; i != v4; v4 = *(_QWORD *)(v4 + 8) )
  {
    v6 = v4 - 56;
    if ( !v4 )
      v6 = 0;
    sub_23BF170(a2, v6);
  }
}
