// Function: sub_5D0DD0
// Address: 0x5d0dd0
//
int __fastcall sub_5D0DD0(__int64 a1, char a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 *v5; // r14
  _QWORD *i; // rbx
  __int64 v7; // rax
  size_t v8; // rdx
  __int64 v9; // rax

  v4 = *(_QWORD *)(a1 + 104);
  if ( v4 )
  {
    v5 = *(__int64 **)(a1 + 104);
    do
    {
      if ( *((_BYTE *)v5 + 8) == 80 )
      {
        for ( i = (_QWORD *)v5[4]; i; i = (_QWORD *)*i )
        {
          v7 = i[5];
          if ( v7 == a3 )
            return v7;
          v8 = *(_QWORD *)(a3 + 176);
          if ( v8 == *(_QWORD *)(v7 + 176) )
          {
            LODWORD(v7) = memcmp(*(const void **)(a3 + 184), *(const void **)(v7 + 184), v8);
            if ( !(_DWORD)v7 )
              return v7;
          }
        }
      }
      v5 = (__int64 *)*v5;
    }
    while ( v5 );
    if ( (*(_BYTE *)(v4 + 11) & 0x20) != 0 )
      goto LABEL_14;
  }
  v4 = sub_727670();
  *(_QWORD *)(v4 + 56) = *(_QWORD *)&dword_4F063F8;
  *(_WORD *)(v4 + 8) = 592;
  v9 = sub_724840(unk_4F073B8, "abi_tag");
  *(_BYTE *)(v4 + 11) |= 0x20u;
  *(_QWORD *)(v4 + 16) = v9;
  *(_QWORD *)v4 = *(_QWORD *)(a1 + 104);
  *(_QWORD *)(a1 + 104) = v4;
LABEL_14:
  v7 = sub_7276D0();
  *(_BYTE *)(v7 + 10) = 3;
  *(_QWORD *)(v7 + 40) = a3;
  *(_QWORD *)v7 = *(_QWORD *)(v4 + 32);
  *(_QWORD *)(v4 + 32) = v7;
  if ( a2 == 11 )
    *(_BYTE *)(a1 + 201) |= 4u;
  else
    *(_BYTE *)(a1 + 168) |= 0x80u;
  return v7;
}
