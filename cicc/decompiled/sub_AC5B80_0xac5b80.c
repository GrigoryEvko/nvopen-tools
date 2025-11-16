// Function: sub_AC5B80
// Address: 0xac5b80
//
__int64 __fastcall sub_AC5B80(__int64 *a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  __int64 result; // rax

  v1 = *a1;
  if ( *a1 )
  {
    v2 = *(_QWORD *)(v1 + 32);
    if ( v2 )
    {
      sub_AC5B80(v2 + 32);
      sub_BD7260(v2);
      sub_BD2DD0(v2);
    }
    sub_BD7260(v1);
    return sub_BD2DD0(v1);
  }
  return result;
}
