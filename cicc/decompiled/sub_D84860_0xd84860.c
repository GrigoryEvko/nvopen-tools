// Function: sub_D84860
// Address: 0xd84860
//
__int64 __fastcall sub_D84860(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 v6; // rdi

  v2 = sub_22077B0(88);
  v3 = v2;
  if ( v2 )
  {
    *(_QWORD *)v2 = a2;
    *(_QWORD *)(v2 + 8) = 0;
    *(_BYTE *)(v2 + 24) = 0;
    *(_BYTE *)(v2 + 40) = 0;
    *(_BYTE *)(v2 + 49) = 0;
    *(_BYTE *)(v2 + 51) = 0;
    *(_QWORD *)(v2 + 56) = 0;
    *(_QWORD *)(v2 + 64) = 0;
    *(_QWORD *)(v2 + 72) = 0;
    *(_DWORD *)(v2 + 80) = 0;
    sub_D84780((__int64 *)v2);
  }
  v4 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v3;
  if ( v4 )
  {
    sub_C7D6A0(*(_QWORD *)(v4 + 64), 16LL * *(unsigned int *)(v4 + 80), 8);
    v5 = *(_QWORD *)(v4 + 8);
    if ( v5 )
    {
      v6 = *(_QWORD *)(v5 + 8);
      if ( v6 )
        j_j___libc_free_0(v6, *(_QWORD *)(v5 + 24) - v6);
      j_j___libc_free_0(v5, 88);
    }
    j_j___libc_free_0(v4, 88);
  }
  return 0;
}
