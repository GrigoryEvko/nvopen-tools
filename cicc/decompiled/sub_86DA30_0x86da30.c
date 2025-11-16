// Function: sub_86DA30
// Address: 0x86da30
//
void __fastcall sub_86DA30(__int64 *a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // r13

  if ( a1 )
  {
    v2 = a1;
    do
    {
      while ( 1 )
      {
        if ( *((_BYTE *)v2 + 8) == 7 )
        {
          v3 = v2[2];
          if ( *(_BYTE *)(v3 + 177) == 2 )
            break;
        }
        v2 = (__int64 *)*v2;
        if ( !v2 )
          return;
      }
      v4 = sub_86B2C0(1);
      *(_QWORD *)(v4 + 40) = a2;
      v5 = v4;
      *(_QWORD *)(v4 + 48) = v3;
      *(_BYTE *)(v4 + 56) = (2 * sub_86D9F0()) | *(_BYTE *)(v4 + 56) & 0xFD;
      sub_86CBE0(v5);
      v2 = (__int64 *)*v2;
    }
    while ( v2 );
  }
}
