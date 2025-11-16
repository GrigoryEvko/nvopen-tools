// Function: sub_886000
// Address: 0x886000
//
void __fastcall sub_886000(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r12

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      v1 = *(_QWORD *)(v1 + 16);
      sub_885FF0(v2, dword_4F04C64, 1);
      *(_BYTE *)(v2 + 81) |= 8u;
    }
    while ( v1 );
  }
}
