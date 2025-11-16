// Function: sub_27C0370
// Address: 0x27c0370
//
void __fastcall sub_27C0370(__int64 *a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 v3; // r13
  __int64 v4; // rdx
  __int64 *i; // rbx
  char v6; // al

  v2 = a1;
  v3 = *a1;
  v4 = *(a1 - 1);
  if ( *a1 != v4 )
  {
    for ( i = a1 - 1; ; --i )
    {
      v2 = i + 1;
      sub_B196A0(*(_QWORD *)(a2 + 16), v3, v4);
      if ( !v6 )
        break;
      v4 = *(i - 1);
      i[1] = *i;
      if ( v3 == v4 )
      {
        *i = v3;
        return;
      }
    }
  }
  *v2 = v3;
}
