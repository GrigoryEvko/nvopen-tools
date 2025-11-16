// Function: sub_C3CCB0
// Address: 0xc3ccb0
//
__int64 __fastcall sub_C3CCB0(__int64 a1)
{
  void *v2; // rbx
  unsigned __int8 *v3; // rdi
  __int64 v4; // rax

  v2 = sub_C33340();
  do
  {
    v3 = *(unsigned __int8 **)(a1 + 8);
    if ( *(void **)v3 == v2 )
      sub_C3CCB0(v3);
    else
      sub_C34440(v3);
    v4 = *(_QWORD *)(a1 + 8);
    a1 = v4 + 24;
  }
  while ( *(void **)(v4 + 24) == v2 );
  return sub_C34440((unsigned __int8 *)(v4 + 24));
}
