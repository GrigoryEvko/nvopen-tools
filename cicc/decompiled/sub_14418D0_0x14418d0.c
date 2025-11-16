// Function: sub_14418D0
// Address: 0x14418d0
//
__int64 __fastcall sub_14418D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v5; // rdi

  v2 = sub_22077B0(56);
  if ( v2 )
  {
    *(_QWORD *)v2 = a2;
    *(_QWORD *)(v2 + 8) = 0;
    *(_BYTE *)(v2 + 24) = 0;
    *(_BYTE *)(v2 + 40) = 0;
    *(_BYTE *)(v2 + 49) = 0;
  }
  v3 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v2;
  if ( v3 )
  {
    v4 = *(_QWORD *)(v3 + 8);
    if ( v4 )
    {
      v5 = *(_QWORD *)(v4 + 8);
      if ( v5 )
        j_j___libc_free_0(v5, *(_QWORD *)(v4 + 24) - v5);
      j_j___libc_free_0(v4, 72);
    }
    j_j___libc_free_0(v3, 56);
  }
  return 0;
}
