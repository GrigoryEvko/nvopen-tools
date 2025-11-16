// Function: sub_733510
// Address: 0x733510
//
void __fastcall sub_733510(__int64 a1)
{
  __int64 v2; // rdi
  _BYTE *v3; // rdx
  __int64 v4; // rax
  bool v5; // zf

  v2 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
  v3 = sub_732EF0(v2);
  v4 = *(_QWORD *)(v2 + 24);
  if ( !v4 )
    v4 = v2 + 32;
  if ( *((_QWORD *)v3 + 19) )
    *(_QWORD *)(*(_QWORD *)(v4 + 56) + 112LL) = a1;
  else
    *((_QWORD *)v3 + 19) = a1;
  *(_QWORD *)(v4 + 56) = a1;
  v5 = *(_QWORD *)(a1 + 40) == 0;
  *(_QWORD *)(a1 + 112) = 0;
  if ( v5 )
    sub_72EE40(a1, 0x2Bu, (__int64)v3);
}
