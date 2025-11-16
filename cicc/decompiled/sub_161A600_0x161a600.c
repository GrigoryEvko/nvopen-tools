// Function: sub_161A600
// Address: 0x161a600
//
__int64 __fastcall sub_161A600(__int64 a1, unsigned __int64 a2)
{
  unsigned int v3; // r12d
  unsigned int v4; // r13d
  __int64 v5; // rdi
  __int64 v6; // rax
  unsigned int v7; // eax
  unsigned int v8; // r12d
  __int64 v9; // rdi

  if ( unk_4F9E388 && !qword_4F9E390 )
  {
    if ( !qword_4F9E2A0 )
      sub_16C1EA0(&qword_4F9E2A0, sub_160CF40, sub_160D890);
    qword_4F9E390 = qword_4F9E2A0;
  }
  v3 = 0;
  v4 = 0;
  sub_1616C40(a1 + 568);
  if ( *(_DWORD *)(a1 + 608) )
  {
    do
    {
      v5 = *(_QWORD *)(*(_QWORD *)(a1 + 600) + 8LL * v3);
      if ( v5 )
        v5 -= 160;
      ++v3;
      v4 |= sub_1619FD0(v5, a2);
      v6 = sub_15E0530(a2);
      sub_16027A0(v6);
      v7 = *(_DWORD *)(a1 + 608);
    }
    while ( v7 > v3 );
    if ( v7 )
    {
      v8 = 0;
      do
      {
        v9 = *(_QWORD *)(*(_QWORD *)(a1 + 600) + 8LL * v8);
        if ( v9 )
          v9 -= 160;
        ++v8;
        sub_160FA30(v9);
      }
      while ( v8 < *(_DWORD *)(a1 + 608) );
    }
  }
  *(_BYTE *)(a1 + 1304) = 1;
  return v4;
}
