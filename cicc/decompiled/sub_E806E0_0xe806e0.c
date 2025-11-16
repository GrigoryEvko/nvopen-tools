// Function: sub_E806E0
// Address: 0xe806e0
//
char __fastcall sub_E806E0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  char result; // al
  __int64 (*v5)(); // rdx

  while ( 2 )
  {
    switch ( *(_BYTE *)a1 )
    {
      case 0:
        result = sub_E806E0(*(_QWORD *)(a1 + 16), a2);
        if ( result )
          return result;
        a1 = *(_QWORD *)(a1 + 24);
        continue;
      case 1:
        return 0;
      case 2:
        v3 = *(_QWORD *)(a1 + 16);
        if ( (*(_BYTE *)(v3 + 9) & 0x70) == 0x20 && *(char *)(v3 + 8) >= 0 )
        {
          *(_BYTE *)(v3 + 8) |= 8u;
          a1 = *(_QWORD *)(v3 + 24);
          continue;
        }
        return v3 == a2;
      case 3:
        a1 = *(_QWORD *)(a1 + 16);
        continue;
      case 4:
        v5 = *(__int64 (**)())(*(_QWORD *)(a1 - 8) + 48LL);
        result = 0;
        if ( v5 != sub_E7FAB0 )
          return ((__int64 (__fastcall *)(__int64, __int64))v5)(a1 - 8, a2);
        return result;
      default:
        BUG();
    }
  }
}
