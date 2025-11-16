// Function: sub_729A40
// Address: 0x729a40
//
void __fastcall sub_729A40(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // r12
  __int64 v4; // r14
  unsigned int v5; // [rsp+Ch] [rbp-24h]

  v3 = a1;
  if ( !a1 )
  {
    v5 = a3;
    v4 = *(_QWORD *)(unk_4F064B0 + 56LL);
    v3 = sub_7B0D00(a2, a3);
    sub_729A00(v4, unk_4F06468);
    a3 = v5;
  }
  sub_729230(v3, a2, a3);
}
