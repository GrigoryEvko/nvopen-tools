// Function: sub_85FC20
// Address: 0x85fc20
//
void __fastcall sub_85FC20(__int64 **a1, int a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  char v4; // dl

  while ( a1 )
  {
    if ( !a2 || ((_BYTE)a1[7] & 0x10) != 0 )
    {
      v3 = (__int64)a1[1];
      v4 = *(_BYTE *)(v3 + 80);
      if ( v4 == 3 || v4 == 2 )
      {
        *(_QWORD *)(v3 + 88) = a1[8];
      }
      else
      {
        v2 = *(_QWORD *)(v3 + 88);
        *(_QWORD *)(v2 + 200) = v3;
        *(_QWORD *)(v2 + 208) = 0;
      }
      *(_BYTE *)(v3 + 82) &= ~1u;
    }
    a1 = (__int64 **)*a1;
  }
}
