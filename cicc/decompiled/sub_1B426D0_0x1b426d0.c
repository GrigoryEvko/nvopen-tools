// Function: sub_1B426D0
// Address: 0x1b426d0
//
__int64 __fastcall sub_1B426D0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  unsigned int v4; // r15d
  __int64 v6; // rcx
  __int64 v8; // r12
  unsigned int v10; // eax
  __int64 v11; // [rsp+8h] [rbp-38h]

  v4 = 1;
  v6 = a1 + 8;
  if ( !a2 && a3 != v6 )
  {
    v10 = sub_16A9900(*a4 + 24LL, (unsigned __int64 *)(*(_QWORD *)(a3 + 32) + 24LL));
    v6 = a1 + 8;
    v4 = v10 >> 31;
  }
  v11 = v6;
  v8 = sub_22077B0(40);
  *(_QWORD *)(v8 + 32) = *a4;
  sub_220F040(v4, v8, a3, v11);
  ++*(_QWORD *)(a1 + 40);
  return v8;
}
