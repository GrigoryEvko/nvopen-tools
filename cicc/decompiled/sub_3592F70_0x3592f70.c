// Function: sub_3592F70
// Address: 0x3592f70
//
void __fastcall sub_3592F70(unsigned __int64 a1)
{
  _QWORD *v2; // rbx
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v2 = *(_QWORD **)(a1 + 280);
  *(_QWORD *)a1 = off_49D8DA8;
  while ( v2 )
  {
    v3 = (unsigned __int64)v2;
    v2 = (_QWORD *)*v2;
    j_j___libc_free_0(v3);
  }
  memset(*(void **)(a1 + 264), 0, 8LL * *(_QWORD *)(a1 + 272));
  v4 = *(_QWORD *)(a1 + 264);
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  if ( v4 != a1 + 312 )
    j_j___libc_free_0(v4);
  sub_C7D6A0(*(_QWORD *)(a1 + 240), (unsigned __int64)*(unsigned int *)(a1 + 256) << 6, 8);
  j_j___libc_free_0(a1);
}
