// Function: sub_1DD91F0
// Address: 0x1dd91f0
//
void __fastcall sub_1DD91F0(__int64 a1, _QWORD *a2)
{
  __int64 *i; // rax
  __int64 v3; // r12
  int *v4; // rax

  if ( (_QWORD *)a1 != a2 )
  {
    for ( i = (__int64 *)a2[11]; i != (__int64 *)a2[12]; i = (__int64 *)a2[11] )
    {
      v3 = *i;
      v4 = (int *)a2[14];
      if ( (int *)a2[15] == v4 )
        sub_1DD8D40(a1, v3);
      else
        sub_1DD8FE0(a1, v3, *v4);
      sub_1DD91B0((__int64)a2, v3);
    }
  }
}
