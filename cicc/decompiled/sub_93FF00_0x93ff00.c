// Function: sub_93FF00
// Address: 0x93ff00
//
void __fastcall sub_93FF00(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx

  sub_93FCC0((__int64)a1, a2);
  v3 = a1[64];
  if ( v3 == a1[65] )
  {
    j_j___libc_free_0(v3, 512);
    v4 = (__int64 *)(a1[67] - 8LL);
    a1[67] = v4;
    v5 = *v4;
    v6 = *v4 + 512;
    a1[65] = v5;
    a1[66] = v6;
    a1[64] = v5 + 504;
    if ( *(_QWORD *)(v5 + 504) )
      sub_B91220(v5 + 504);
  }
  else
  {
    a1[64] = v3 - 8;
    if ( *(_QWORD *)(v3 - 8) )
      sub_B91220(v3 - 8);
  }
}
