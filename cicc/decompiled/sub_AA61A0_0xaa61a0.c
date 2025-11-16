// Function: sub_AA61A0
// Address: 0xaa61a0
//
void __fastcall sub_AA61A0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // r14
  __int64 v5; // r13
  _QWORD *v6; // rax
  __int64 v7; // rax
  __int64 v8; // r12
  _QWORD *v9; // rax

  if ( a4 )
  {
    v4 = *(_QWORD *)(a3 + 16);
    v5 = *(_QWORD *)(v4 + 16);
    if ( v5 != a3 )
    {
      v6 = sub_AA4580(a1, a2);
      sub_B14570(v6, v5, a3, v4, 1);
    }
  }
  else
  {
    v7 = sub_AA6190(a1, a2);
    v8 = v7;
    if ( v7 && (*(_QWORD *)(v7 + 8) & 0xFFFFFFFFFFFFFFF8LL) != v7 + 8 )
    {
      v9 = sub_AA4580(a1, a2);
      sub_B14410(v9, v8, 0);
    }
  }
}
