// Function: sub_157F980
// Address: 0x157f980
//
__int64 __fastcall sub_157F980(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r13
  __int64 v3; // r14
  __int64 v4; // rax
  unsigned __int64 *v5; // rcx
  unsigned __int64 v6; // rdx

  v1 = *(_QWORD *)(a1 + 56);
  v2 = *(_QWORD *)(a1 + 32);
  sub_157F970(a1, 0);
  if ( (*(_BYTE *)(a1 + 23) & 0x20) != 0 )
  {
    v3 = *(_QWORD *)(v1 + 104);
    if ( v3 )
    {
      v4 = sub_16498B0(a1);
      sub_164D860(v3, v4);
    }
  }
  v5 = *(unsigned __int64 **)(a1 + 32);
  v6 = *(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  *v5 = v6 | *v5 & 7;
  *(_QWORD *)(v6 + 8) = v5;
  *(_QWORD *)(a1 + 24) &= 7uLL;
  *(_QWORD *)(a1 + 32) = 0;
  sub_157EF40(a1);
  j_j___libc_free_0(a1, 64);
  return v2;
}
