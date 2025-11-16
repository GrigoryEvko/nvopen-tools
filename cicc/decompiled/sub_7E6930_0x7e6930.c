// Function: sub_7E6930
// Address: 0x7e6930
//
void __fastcall sub_7E6930(__int64 *a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v3; // r12
  _BYTE *v4; // r14
  __int64 v5; // rax

  if ( a1 )
  {
    v2 = a1;
    do
    {
      if ( *((_BYTE *)v2 + 8) == 7 )
      {
        v3 = v2[2];
        if ( *(_BYTE *)(v3 + 177) == 2 )
        {
          v4 = sub_726B30(17);
          v5 = *(_QWORD *)(v3 + 184);
          *((_QWORD *)v4 + 9) = v5;
          *(_BYTE *)(v5 + 49) |= 2u;
          sub_7E6810((__int64)v4, a2, 1);
          sub_804750(v4, a2);
        }
        if ( (*(_BYTE *)(v3 + 170) & 2) != 0 )
          sub_7E6930(*(_QWORD *)(v3 + 128), a2);
      }
      v2 = (__int64 *)*v2;
    }
    while ( v2 );
  }
}
