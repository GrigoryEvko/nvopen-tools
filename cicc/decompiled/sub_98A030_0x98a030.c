// Function: sub_98A030
// Address: 0x98a030
//
__int64 __fastcall sub_98A030(unsigned int a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v7; // rcx
  _BYTE *v9; // rax

  if ( *(_BYTE *)a4 == 18 )
  {
    v7 = a4 + 24;
    return sub_989FD0(a1, a2, a3, v7, a5);
  }
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a4 + 8) + 8LL) - 17 <= 1 && *(_BYTE *)a4 <= 0x15u )
  {
    v9 = (_BYTE *)sub_AD7630(a4, 1);
    if ( v9 )
    {
      if ( *v9 == 18 )
      {
        v7 = (__int64)(v9 + 24);
        return sub_989FD0(a1, a2, a3, v7, a5);
      }
    }
  }
  return 0;
}
