// Function: sub_2231430
// Address: 0x2231430
//
__int64 __fastcall sub_2231430(__int64 a1)
{
  unsigned __int64 v1; // rax
  unsigned int v2; // r8d
  __int64 v3; // rdi

  v1 = sub_22091A0(&qword_4FD69B0);
  v2 = 0;
  if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) > v1 )
  {
    v3 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 8 * v1);
    if ( v3 )
      LOBYTE(v2) = sub_2252480(
                     v3,
                     &`typeinfo for'std::locale::facet,
                     &`typeinfo for'std::num_get<char,std::istreambuf_iterator<char>>,
                     0) != 0;
  }
  return v2;
}
