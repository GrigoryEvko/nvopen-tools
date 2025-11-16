// Function: sub_1357E10
// Address: 0x1357e10
//
__int64 __fastcall sub_1357E10(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  int v4; // eax
  int v5; // edx
  unsigned __int64 *v6; // rcx
  unsigned __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // rax

  v3 = *(_QWORD *)(a2 + 32);
  if ( v3 )
  {
    v4 = *(_DWORD *)(v3 + 64);
    v5 = (v4 + 0x7FFFFFF) & 0x7FFFFFF;
    *(_DWORD *)(v3 + 64) = v5 | v4 & 0xF8000000;
    if ( !v5 )
      sub_1357730(v3, a1);
    *(_QWORD *)(a2 + 32) = 0;
  }
  if ( (*(_BYTE *)(a2 + 67) & 0x40) != 0 )
    *(_DWORD *)(a1 + 56) -= *(_DWORD *)(a2 + 68);
  v6 = *(unsigned __int64 **)(a2 + 8);
  v7 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
  *v6 = v7 | *v6 & 7;
  *(_QWORD *)(v7 + 8) = v6;
  v8 = *(_QWORD *)(a2 + 48);
  v9 = *(_QWORD *)(a2 + 40);
  *(_QWORD *)a2 &= 7uLL;
  *(_QWORD *)(a2 + 8) = 0;
  if ( v8 != v9 )
  {
    do
    {
      v10 = *(_QWORD *)(v9 + 16);
      if ( v10 != 0 && v10 != -8 && v10 != -16 )
        sub_1649B30(v9);
      v9 += 24;
    }
    while ( v8 != v9 );
    v9 = *(_QWORD *)(a2 + 40);
  }
  if ( v9 )
    j_j___libc_free_0(v9, *(_QWORD *)(a2 + 56) - v9);
  return j_j___libc_free_0(a2, 72);
}
