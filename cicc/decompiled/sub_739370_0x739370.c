// Function: sub_739370
// Address: 0x739370
//
__int64 __fastcall sub_739370(__int128 a1, unsigned int a2)
{
  __int64 v3; // r12
  bool v4; // dl
  __int64 v5; // rbx
  __int128 v6; // rdi

  v3 = *((_QWORD *)&a1 + 1);
  v4 = *((_QWORD *)&a1 + 1) == 0;
  if ( a1 == 0 )
    return 1;
  if ( (_QWORD)a1 )
  {
    v5 = a1;
    do
    {
      if ( v4 )
        break;
      *((_QWORD *)&v6 + 1) = v3;
      *(_QWORD *)&v6 = v5;
      if ( !(unsigned int)sub_7386E0(v6, a2) )
        break;
      v5 = *(_QWORD *)(v5 + 16);
      v3 = *(_QWORD *)(v3 + 16);
      v4 = v3 == 0;
      if ( !(v3 | v5) )
        return 1;
    }
    while ( v5 );
  }
  return 0;
}
