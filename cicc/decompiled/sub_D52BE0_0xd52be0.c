// Function: sub_D52BE0
// Address: 0xd52be0
//
__int64 __fastcall sub_D52BE0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rbx
  unsigned int v4; // r12d
  __int64 v5; // rdi

  v2 = *(__int64 **)(a1 + 8);
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v2 == 8 )
  {
    v3 = a1;
    v4 = 1;
    do
    {
      v5 = v3;
      v3 = *v2;
      if ( !sub_D52BD0(v5, (_QWORD *)*v2, a2) )
        break;
      v2 = *(__int64 **)(v3 + 8);
      ++v4;
    }
    while ( *(_QWORD *)(v3 + 16) - (_QWORD)v2 == 8 );
  }
  else
  {
    return 1;
  }
  return v4;
}
