// Function: sub_349D190
// Address: 0x349d190
//
void __fastcall sub_349D190(unsigned __int64 a1)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdi

  v2 = *(_QWORD *)(a1 + 248);
  *(_QWORD *)a1 = off_4A375B0;
  v3 = v2 + 8LL * *(unsigned int *)(a1 + 256);
  if ( v2 != v3 )
  {
    do
    {
      v4 = *(_QWORD *)(v3 - 8);
      v3 -= 8LL;
      if ( v4 )
      {
        v5 = *(_QWORD *)(v4 + 24);
        if ( v5 != v4 + 40 )
          _libc_free(v5);
        j_j___libc_free_0(v4);
      }
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)(a1 + 248);
  }
  if ( v3 != a1 + 264 )
    _libc_free(v3);
  v6 = *(_QWORD *)(a1 + 224);
  if ( v6 != a1 + 240 )
    _libc_free(v6);
  v7 = *(_QWORD *)(a1 + 208);
  if ( v7 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 16LL))(v7);
  v8 = *(_QWORD *)(a1 + 200);
  if ( v8 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v8 + 16LL))(v8);
  *(_QWORD *)a1 = &unk_49DAF80;
  sub_BB9100(a1);
  j_j___libc_free_0(a1);
}
