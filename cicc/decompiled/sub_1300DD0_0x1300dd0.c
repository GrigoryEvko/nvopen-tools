// Function: sub_1300DD0
// Address: 0x1300dd0
//
__int64 __fastcall sub_1300DD0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r12
  unsigned int v3; // eax
  __int64 v4; // rdx
  __int64 *v5; // rax
  __int64 *v6; // rbx
  __int64 *v7; // r13

  v2 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
  {
    a2 = 0;
    v2 = sub_1313D30(v2, 0);
  }
  sub_1314090(v2);
  nullsub_509(v2 + 2616);
  sub_130F980(v2);
  v3 = sub_1300B70();
  v4 = v3;
  if ( v3 )
  {
    v5 = qword_50579C0;
    v4 = (unsigned int)(v4 - 1);
    v6 = &qword_50579C0[1];
    v7 = &qword_50579C0[(unsigned int)v4 + 1];
    while ( 1 )
    {
      a2 = *v5;
      if ( *v5 )
        sub_1319280(v2);
      v5 = v6;
      if ( v6 == v7 )
        break;
      ++v6;
    }
  }
  nullsub_505(v2, a2, v4);
  sub_131A8A0(v2);
  sub_130B050(v2, &unk_5057920);
  sub_1313250(v2);
  return sub_133D740(v2);
}
