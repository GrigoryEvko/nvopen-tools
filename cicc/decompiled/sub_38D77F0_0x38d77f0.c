// Function: sub_38D77F0
// Address: 0x38d77f0
//
void __fastcall sub_38D77F0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 *v3; // rbx
  unsigned __int64 *v4; // rdi
  unsigned __int64 v5; // rdx

  *a1 = &unk_4A3E548;
  v2 = a1[14];
  if ( (_QWORD *)v2 != a1 + 16 )
    _libc_free(v2);
  v3 = (unsigned __int64 *)a1[13];
  while ( a1 + 12 != v3 )
  {
    v4 = v3;
    v3 = (unsigned __int64 *)v3[1];
    v5 = *v4 & 0xFFFFFFFFFFFFFFF8LL;
    *v3 = v5 | *v3 & 7;
    *(_QWORD *)(v5 + 8) = v3;
    *v4 &= 7u;
    v4[1] = 0;
    sub_38CFA90((unsigned __int64)v4);
  }
  nullsub_1930();
}
