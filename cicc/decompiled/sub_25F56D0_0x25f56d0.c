// Function: sub_25F56D0
// Address: 0x25f56d0
//
__int64 __fastcall sub_25F56D0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r15
  __int64 v4; // r8
  __int64 v5; // r14
  char i; // [rsp+Eh] [rbp-32h]
  unsigned __int8 v8; // [rsp+Fh] [rbp-31h]

  v2 = sub_BAA6A0(a2, 0);
  v3 = *(_QWORD *)(a2 + 32);
  v8 = 0;
  for ( i = v2 != 0; a2 + 24 != v3; v3 = *(_QWORD *)(v3 + 8) )
  {
    v4 = v3 - 56;
    if ( !v3 )
      v4 = 0;
    v5 = v4;
    if ( !sub_B2FC80(v4) && !(unsigned __int8)sub_B2D610(v5, 48) )
    {
      if ( sub_25F03E0(a1, v5) )
      {
        v8 |= sub_25EFA30(v5, 0);
      }
      else if ( sub_25F0850((__int64)a1, v5) )
      {
        v8 |= sub_25F3660((__int64)a1, v5, i);
      }
    }
  }
  return v8;
}
