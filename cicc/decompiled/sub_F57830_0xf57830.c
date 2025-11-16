// Function: sub_F57830
// Address: 0xf57830
//
void __fastcall sub_F57830(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  char v5; // al
  unsigned __int64 v6; // rax
  __int64 v7; // r14
  unsigned __int8 *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_QWORD *)(a3 + 8);
  v5 = *(_BYTE *)(v4 + 8);
  if ( v5 == 14 )
  {
    sub_B99FD0(a3, 0xBu, a2);
  }
  else if ( v5 == 12 )
  {
    v12[0] = sub_BD5C60(a3);
    v6 = sub_AC9EC0(*(__int64 ***)(*(_QWORD *)(a1 - 32) + 8LL));
    v7 = sub_AD4C50(v6, (__int64 **)v4, 0);
    v8 = (unsigned __int8 *)sub_ACD640(v4, 1, 0);
    v9 = sub_AD57C0(v7, v8, 0, 0);
    v11 = sub_B8C7C0(v12, v9, v7, v10);
    sub_B99FD0(a3, 4u, v11);
  }
}
