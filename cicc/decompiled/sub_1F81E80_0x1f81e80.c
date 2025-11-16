// Function: sub_1F81E80
// Address: 0x1f81e80
//
void __fastcall sub_1F81E80(__int64 *a1, __int64 a2)
{
  unsigned int v3; // edx
  __int64 *v4; // rbx
  __int64 *i; // r13
  __int64 v6; // rsi
  __int64 v7; // rax

  sub_1F6D2A0((__int64)a1, a2);
  v4 = *(__int64 **)(a2 + 32);
  for ( i = &v4[5 * *(unsigned int *)(a2 + 56)]; i != v4; v4 += 5 )
  {
    v6 = *v4;
    v7 = *(_QWORD *)(*v4 + 48);
    if ( v7 && !*(_QWORD *)(v7 + 32) || *(_DWORD *)(v6 + 60) > 1u )
      sub_1F81BC0((__int64)a1, v6);
  }
  sub_1D2DE10(*a1, a2, v3);
}
