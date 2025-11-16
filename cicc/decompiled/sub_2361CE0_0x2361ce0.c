// Function: sub_2361CE0
// Address: 0x2361ce0
//
void __fastcall sub_2361CE0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // r13
  _QWORD *v5; // rax
  char *v6; // rsi
  _BYTE *v7; // rsi
  _QWORD v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = sub_BC1CD0(a2, &unk_4F86710, a1) + 8;
  v5 = (_QWORD *)sub_22077B0(0x10u);
  if ( v5 )
  {
    v5[1] = v4;
    *v5 = &unk_49DD930;
  }
  v8[0] = v5;
  v6 = (char *)a3[2];
  if ( v6 != (char *)a3[3] )
  {
    if ( v6 )
    {
      *(_QWORD *)v6 = v5;
      v6 = (char *)a3[2];
    }
    v8[0] = &unk_4F86710;
    a3[2] = (__int64)(v6 + 8);
    v7 = (_BYTE *)a3[5];
    if ( v7 != (_BYTE *)a3[6] )
      goto LABEL_7;
LABEL_11:
    sub_234F040((__int64)(a3 + 4), v7, v8);
    return;
  }
  sub_CF7870(a3 + 1, v6, v8);
  v8[0] = &unk_4F86710;
  v7 = (_BYTE *)a3[5];
  if ( v7 == (_BYTE *)a3[6] )
    goto LABEL_11;
LABEL_7:
  if ( v7 )
  {
    *(_QWORD *)v7 = &unk_4F86710;
    v7 = (_BYTE *)a3[5];
  }
  a3[5] = (__int64)(v7 + 8);
}
