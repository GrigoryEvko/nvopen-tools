// Function: sub_129F180
// Address: 0x129f180
//
void __fastcall sub_129F180(_QWORD *a1, _QWORD *a2)
{
  __int64 v3; // rdi
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx

  sub_129F080((__int64)a1, a2);
  v3 = a1[68];
  if ( v3 == a1[69] )
  {
    j_j___libc_free_0(v3, 512);
    v4 = (__int64 *)(a1[71] - 8LL);
    a1[71] = v4;
    v5 = *v4;
    v6 = *v4 + 512;
    a1[69] = v5;
    a1[70] = v6;
    a1[68] = v5 + 504;
    if ( *(_QWORD *)(v5 + 504) )
      sub_161E7C0(v5 + 504);
  }
  else
  {
    a1[68] = v3 - 8;
    if ( *(_QWORD *)(v3 - 8) )
      sub_161E7C0(v3 - 8);
  }
}
