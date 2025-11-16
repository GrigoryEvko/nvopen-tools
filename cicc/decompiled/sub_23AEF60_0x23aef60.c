// Function: sub_23AEF60
// Address: 0x23aef60
//
void __fastcall sub_23AEF60(_BYTE *a1, __int64 a2, __m128i a3)
{
  __int64 i; // rbx
  __int64 v4; // r12
  char *v5; // rax
  __int64 v6; // rdx

  if ( sub_BC63A0("*", 1) || (unsigned __int8)sub_BC5DE0() )
  {
    sub_A69980((__int64 (__fastcall **)())a2, (__int64)a1, 0, 0, 0, a3);
  }
  else
  {
    for ( i = *(_QWORD *)(a2 + 32); a2 + 24 != i; i = *(_QWORD *)(i + 8) )
    {
      v4 = i - 56;
      if ( !i )
        v4 = 0;
      v5 = (char *)sub_BD5D20(v4);
      if ( sub_BC63A0(v5, v6) )
        sub_A69870(v4, a1, 0);
    }
  }
}
