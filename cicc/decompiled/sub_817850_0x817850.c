// Function: sub_817850
// Address: 0x817850
//
void __fastcall sub_817850(__int64 a1, unsigned int a2, __int64 *a3)
{
  __int64 v4; // rbx

  if ( a1 )
  {
    v4 = a1;
    do
    {
      if ( (*(_BYTE *)(v4 + 25) & 0x10) != 0 )
        break;
      if ( (*(_BYTE *)(v4 + 26) & 4) != 0 )
      {
        *a3 += 2;
        sub_8238B0(qword_4F18BE0, "sp", 2);
      }
      sub_816460(v4, a2, 0, a3);
      v4 = *(_QWORD *)(v4 + 16);
    }
    while ( v4 );
  }
}
