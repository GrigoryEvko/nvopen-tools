// Function: sub_7DA010
// Address: 0x7da010
//
void __fastcall sub_7DA010(__int64 a1)
{
  __int64 v1; // rbx

  v1 = *(_QWORD *)(a1 + 72);
  sub_7D9F20(v1);
  if ( !*(_BYTE *)(v1 + 32) )
    *(_QWORD *)(v1 + 48) = sub_7E7ED0(*(_QWORD *)(v1 + 16));
}
