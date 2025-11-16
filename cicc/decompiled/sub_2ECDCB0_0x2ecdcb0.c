// Function: sub_2ECDCB0
// Address: 0x2ecdcb0
//
void __fastcall sub_2ECDCB0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned __int64 v4; // r13
  _QWORD *v6; // rax
  unsigned __int64 i; // rax
  __int64 *v8; // r12

  if ( a2 != a3 )
  {
    v4 = a4;
    v6 = (_QWORD *)sub_22077B0(0x20u);
    v6[2] = a2;
    v6[3] = a3;
    sub_2208C80(v6, a1);
    ++*(_QWORD *)(a1 + 16);
    sub_2ECDC20(a1);
    for ( i = *(_QWORD *)(a1 + 16); i > v4; i = *(_QWORD *)(a1 + 16) )
    {
      v8 = *(__int64 **)a1;
      *(_QWORD *)(a1 + 16) = i - 1;
      sub_2208CA0(v8);
      j_j___libc_free_0((unsigned __int64)v8);
    }
  }
}
