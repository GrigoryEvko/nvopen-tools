// Function: sub_2306F40
// Address: 0x2306f40
//
__int64 __fastcall sub_2306F40(__int64 a1)
{
  __int64 v2; // rsi
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  unsigned __int64 v5; // rdi

  v2 = *(unsigned int *)(a1 + 32);
  *(_QWORD *)a1 = &unk_4A0B290;
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 16);
    v4 = &v3[4 * v2];
    do
    {
      if ( *v3 != -8192 && *v3 != -4096 )
      {
        v5 = v3[1];
        if ( v5 )
          j_j___libc_free_0(v5);
      }
      v3 += 4;
    }
    while ( v4 != v3 );
    v2 = *(unsigned int *)(a1 + 32);
  }
  return sub_C7D6A0(*(_QWORD *)(a1 + 16), 32 * v2, 8);
}
