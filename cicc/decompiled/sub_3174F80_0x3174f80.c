// Function: sub_3174F80
// Address: 0x3174f80
//
void __fastcall sub_3174F80(__int64 a1)
{
  __int64 i; // r14
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rdx
  _QWORD *v5; // r15

  for ( i = *(_QWORD *)(a1 + 80); a1 + 72 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    v2 = *(_QWORD *)(i + 32);
    while ( i + 24 != v2 )
    {
      v3 = v2;
      v2 = *(_QWORD *)(v2 + 8);
      if ( *(_BYTE *)(v3 - 24) == 85 )
      {
        v4 = *(_QWORD *)(v3 - 56);
        if ( v4 )
        {
          if ( !*(_BYTE *)v4
            && *(_QWORD *)(v4 + 24) == *(_QWORD *)(v3 + 56)
            && (*(_BYTE *)(v4 + 33) & 0x20) != 0
            && *(_DWORD *)(v4 + 36) == 336 )
          {
            v5 = (_QWORD *)(v3 - 24);
            sub_BD84D0(v3 - 24, *(_QWORD *)(v3 - 32LL * (*(_DWORD *)(v3 - 20) & 0x7FFFFFF) - 24));
            sub_B43D60(v5);
          }
        }
      }
    }
  }
}
