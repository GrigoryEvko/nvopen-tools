// Function: sub_92FEA0
// Address: 0x92fea0
//
__int64 __fastcall sub_92FEA0(__int64 a1, _QWORD *a2, char a3)
{
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rsi

  sub_92FD90(a1, (__int64)a2);
  if ( a3 && !a2[2] )
  {
    sub_AA5290(a2);
    return j_j___libc_free_0(a2, 80);
  }
  else
  {
    v5 = *(_QWORD *)(a1 + 192);
    sub_B2B790(v5 + 72, a2);
    v6 = a2[3];
    v7 = *(_QWORD *)(v5 + 72);
    a2[4] = v5 + 72;
    v7 &= 0xFFFFFFFFFFFFFFF8LL;
    a2[3] = v7 | v6 & 7;
    *(_QWORD *)(v7 + 8) = a2 + 3;
    v8 = *(unsigned __int8 *)(v5 + 128);
    *(_QWORD *)(v5 + 72) = *(_QWORD *)(v5 + 72) & 7LL | (unsigned __int64)(a2 + 3);
    sub_AA4C30(a2, v8);
    *(_QWORD *)(a1 + 96) = a2;
    *(_QWORD *)(a1 + 104) = a2 + 6;
    *(_WORD *)(a1 + 112) = 0;
    return 0;
  }
}
