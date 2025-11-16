// Function: sub_AA6320
// Address: 0xaa6320
//
void __fastcall sub_AA6320(__int64 a1)
{
  unsigned __int64 v1; // rbx
  __int64 v2; // r13

  if ( *(_BYTE *)(a1 + 40) )
  {
    v1 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v1 != a1 + 48 )
    {
      if ( !v1 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v1 - 24) - 30 <= 0xA )
      {
        v2 = sub_AA60B0(a1);
        if ( v2 )
        {
          sub_AA4580(a1, v1 - 24);
          sub_B14410(*(_QWORD *)(v1 + 40), v2, 0);
          sub_B14200(v2);
          sub_AA6260(a1);
        }
      }
    }
  }
}
