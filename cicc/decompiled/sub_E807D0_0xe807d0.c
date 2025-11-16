// Function: sub_E807D0
// Address: 0xe807d0
//
void *__fastcall sub_E807D0(__int64 a1)
{
  __int64 v2; // rbx
  void *v3; // r12
  __int64 v4; // rax
  void *v6; // rax

  while ( 2 )
  {
    switch ( *(_BYTE *)a1 )
    {
      case 0:
        v3 = (void *)sub_E807D0(*(_QWORD *)(a1 + 16));
        v6 = (void *)sub_E807D0(*(_QWORD *)(a1 + 24));
        if ( off_4C5D170 == v3 )
          return v6;
        if ( off_4C5D170 == v6 )
          return v3;
        if ( *(_DWORD *)a1 >> 8 == 18 )
          return off_4C5D170;
        if ( !v3 )
          return v6;
        return v3;
      case 1:
        return off_4C5D170;
      case 2:
        v2 = *(_QWORD *)(a1 + 16);
        v3 = *(void **)v2;
        if ( !*(_QWORD *)v2 && (*(_BYTE *)(v2 + 9) & 0x70) == 0x20 && *(char *)(v2 + 8) >= 0 )
        {
          *(_BYTE *)(v2 + 8) |= 8u;
          v4 = sub_E807D0(*(_QWORD *)(v2 + 24));
          *(_QWORD *)v2 = v4;
          return (void *)v4;
        }
        return v3;
      case 3:
        a1 = *(_QWORD *)(a1 + 16);
        continue;
      case 4:
        return (void *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)(a1 - 8) + 72LL))(a1 - 8);
      default:
        BUG();
    }
  }
}
