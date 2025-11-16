// Function: sub_733310
// Address: 0x733310
//
void __fastcall sub_733310(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 v4; // rdi
  __int64 v5; // rbx
  _BYTE *v6; // rdx
  __int64 v7; // rax
  bool v8; // zf

  v3 = qword_4F04C68[0];
  if ( !a2 )
    v3 = 776LL * (int)dword_4F04C5C + qword_4F04C68[0];
  v4 = v3;
  v5 = v3 + 32;
  v6 = sub_732EF0(v4);
  v7 = *(_QWORD *)(v5 - 8);
  if ( !v7 )
    v7 = v5;
  if ( *((_QWORD *)v6 + 12) )
    *(_QWORD *)(*(_QWORD *)(v7 + 24) + 120LL) = a1;
  else
    *((_QWORD *)v6 + 12) = a1;
  *(_QWORD *)(v7 + 24) = a1;
  v8 = *(_QWORD *)(a1 + 40) == 0;
  *(_QWORD *)(a1 + 120) = 0;
  if ( v8 && (*(_BYTE *)(a1 + 89) & 2) == 0 )
    sub_72EE40(a1, 2u, (__int64)v6);
}
