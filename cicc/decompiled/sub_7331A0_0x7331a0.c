// Function: sub_7331A0
// Address: 0x7331a0
//
void __fastcall sub_7331A0(__int64 a1)
{
  __int64 v2; // rdi
  _BYTE *v3; // rdx
  __int64 v4; // rax

  v2 = qword_4F04C68[0] + 776LL * dword_4F04C34;
  v3 = sub_732EF0(v2);
  v4 = *(_QWORD *)(v2 + 24);
  if ( !v4 )
    v4 = v2 + 32;
  if ( *((_QWORD *)v3 + 21) )
  {
    *(_QWORD *)(*(_QWORD *)(v4 + 72) + 112LL) = a1;
    *(_QWORD *)(v4 + 72) = a1;
    if ( *(_QWORD *)(a1 + 40) )
      return;
LABEL_7:
    sub_72EE40(a1, 0x1Cu, (__int64)v3);
    return;
  }
  *((_QWORD *)v3 + 21) = a1;
  *(_QWORD *)(v4 + 72) = a1;
  if ( !*(_QWORD *)(a1 + 40) )
    goto LABEL_7;
}
