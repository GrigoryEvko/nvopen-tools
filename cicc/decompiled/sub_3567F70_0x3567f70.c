// Function: sub_3567F70
// Address: 0x3567f70
//
__int64 __fastcall sub_3567F70(_QWORD *a1)
{
  __int64 v1; // r12
  __int64 *v2; // rbx
  __int64 *v3; // r14
  __int64 v4; // r15
  char v5; // al

  v1 = a1[4];
  if ( v1 )
  {
    v2 = *(__int64 **)(v1 + 64);
    v3 = &v2[*(unsigned int *)(v1 + 72)];
    if ( v2 == v3 )
    {
      return 0;
    }
    else
    {
      v1 = 0;
      do
      {
        v4 = *v2;
        v5 = sub_3567D90(a1, *v2);
        if ( v4 && v5 )
        {
          if ( v1 )
            return 0;
          v1 = v4;
        }
        ++v2;
      }
      while ( v3 != v2 );
    }
  }
  return v1;
}
