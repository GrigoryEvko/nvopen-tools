// Function: sub_72FBE0
// Address: 0x72fbe0
//
void __fastcall sub_72FBE0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx

  v1 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
  v2 = *(_QWORD *)(v1 + 184);
  if ( *(_QWORD *)(v2 + 40) )
    *(_QWORD *)(*(_QWORD *)(v1 + 280) + 112LL) = a1;
  else
    *(_QWORD *)(v2 + 40) = a1;
  *(_QWORD *)(v1 + 280) = a1;
  *(_QWORD *)(a1 + 112) = 0;
  sub_72EE40(a1, 7u, v2);
}
