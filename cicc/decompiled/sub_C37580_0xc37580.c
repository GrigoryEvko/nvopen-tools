// Function: sub_C37580
// Address: 0xc37580
//
__int64 __fastcall sub_C37580(__int64 a1, __int64 a2)
{
  char *v2; // rax
  char v3; // al
  int v4; // eax
  unsigned int v5; // r12d
  unsigned int v7; // r14d
  __int64 v8; // r12
  __int64 v9; // rax
  _QWORD v10[4]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v11[8]; // [rsp+20h] [rbp-40h] BYREF

  v2 = (char *)sub_C94E20(qword_4F863F0);
  if ( v2 )
    v3 = *v2;
  else
    v3 = qword_4F863F0[2];
  if ( v3 && *(_UNKNOWN **)a1 == &unk_3F657C0 && (sub_C33940(a1) || sub_C33940(a2)) )
  {
    sub_C33EB0(v10, (__int64 *)a1);
    sub_C33EB0(v11, (__int64 *)a2);
    if ( sub_C33940((__int64)v10) )
      sub_C37310((__int64)v10, 0);
    if ( sub_C33940((__int64)v11) )
      sub_C37310((__int64)v11, 0);
    v5 = sub_C37580(v10, v11);
    sub_C338F0((__int64)v11);
    sub_C338F0((__int64)v10);
  }
  else
  {
    v4 = *(_DWORD *)(a1 + 16) - *(_DWORD *)(a2 + 16);
    if ( !v4 )
    {
      v7 = sub_C337D0(a1);
      v8 = sub_C33930(a2);
      v9 = sub_C33930(a1);
      v4 = sub_C49940(v9, v8, v7);
    }
    v5 = v4 == 0;
    if ( v4 > 0 )
      return 2;
  }
  return v5;
}
