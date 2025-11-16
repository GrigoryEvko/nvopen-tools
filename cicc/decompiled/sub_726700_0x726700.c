// Function: sub_726700
// Address: 0x726700
//
void *__fastcall sub_726700(int a1)
{
  void *v1; // r12

  if ( dword_4F07270[0] == unk_4F073B8 )
  {
    v1 = (void *)qword_4F06BB0;
    if ( qword_4F06BB0 )
      qword_4F06BB0 = *(_QWORD *)(qword_4F06BB0 + 80LL);
    else
      v1 = sub_7247C0(88);
  }
  else
  {
    v1 = sub_7246D0(88);
  }
  sub_7266C0((__int64)v1, a1);
  return v1;
}
