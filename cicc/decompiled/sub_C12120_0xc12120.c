// Function: sub_C12120
// Address: 0xc12120
//
__int64 __fastcall sub_C12120(__int64 a1)
{
  _QWORD *v1; // r13
  _QWORD *v2; // r12

  v1 = *(_QWORD **)(a1 + 232);
  v2 = *(_QWORD **)(a1 + 224);
  *(_QWORD *)a1 = &unk_49E3560;
  if ( v1 != v2 )
  {
    do
    {
      if ( *v2 )
        j_j___libc_free_0(*v2, v2[2] - *v2);
      v2 += 3;
    }
    while ( v1 != v2 );
    v2 = *(_QWORD **)(a1 + 224);
  }
  if ( v2 )
    j_j___libc_free_0(v2, *(_QWORD *)(a1 + 240) - (_QWORD)v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 200), 8LL * *(unsigned int *)(a1 + 216), 4);
  return sub_C7D6A0(*(_QWORD *)(a1 + 168), 8LL * *(unsigned int *)(a1 + 184), 4);
}
