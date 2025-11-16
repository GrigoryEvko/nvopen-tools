// Function: sub_38CE440
// Address: 0x38ce440
//
void *__fastcall sub_38CE440(__int64 a1)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r12
  void *v5; // rax

  while ( 2 )
  {
    switch ( *(_DWORD *)a1 )
    {
      case 0:
        v3 = sub_38CE440(*(_QWORD *)(a1 + 24));
        v5 = (void *)sub_38CE440(*(_QWORD *)(a1 + 32));
        if ( off_4CF6DB8 == (_UNKNOWN *)v3 )
          return v5;
        if ( off_4CF6DB8 == v5 )
          return (void *)v3;
        if ( *(_DWORD *)(a1 + 16) == 17 )
          return off_4CF6DB8;
        if ( !v3 )
          return v5;
        return (void *)v3;
      case 1:
        return off_4CF6DB8;
      case 2:
        v2 = *(_QWORD *)(a1 + 24);
        v3 = *(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v3 && (*(_BYTE *)(v2 + 9) & 0xC) == 8 )
        {
          *(_BYTE *)(v2 + 8) |= 4u;
          v3 = sub_38CE440(*(_QWORD *)(v2 + 24));
          *(_QWORD *)v2 = v3 | *(_QWORD *)v2 & 7LL;
        }
        return (void *)v3;
      case 3:
        a1 = *(_QWORD *)(a1 + 24);
        continue;
      case 4:
        return (void *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)(a1 - 8) + 64LL))(a1 - 8);
    }
  }
}
