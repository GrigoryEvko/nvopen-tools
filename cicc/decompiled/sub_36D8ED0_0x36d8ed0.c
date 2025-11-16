// Function: sub_36D8ED0
// Address: 0x36d8ed0
//
void *__fastcall sub_36D8ED0(_QWORD *a1, __int64 a2, int a3)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = sub_22077B0(0x478u);
  v5 = v4;
  if ( v4 )
    sub_36D8E00(v4, a2, a3);
  v7[0] = v5;
  sub_3420A50((__int64)a1, (__int64)&unk_5040DC8, v7);
  if ( v7[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v7[0] + 8LL))(v7[0]);
  *a1 = &unk_4A3BDD0;
  return &unk_4A3BDD0;
}
