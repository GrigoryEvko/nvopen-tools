// Function: sub_100ACE0
// Address: 0x100ace0
//
bool __fastcall sub_100ACE0(__int64 a1, int a2, unsigned __int8 *a3)
{
  __int64 v5; // rdi
  __int64 v7; // rdx
  _BYTE *v8; // rax

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = *((_QWORD *)a3 - 8);
  if ( *(_BYTE *)v5 == 17 )
  {
    **(_QWORD **)a1 = v5 + 24;
    return *((_QWORD *)a3 - 4) == *(_QWORD *)(a1 + 16);
  }
  v7 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 8LL) - 17;
  if ( (unsigned int)v7 <= 1 && *(_BYTE *)v5 <= 0x15u )
  {
    v8 = sub_AD7630(v5, *(unsigned __int8 *)(a1 + 8), v7);
    if ( v8 )
    {
      if ( *v8 == 17 )
      {
        **(_QWORD **)a1 = v8 + 24;
        return *((_QWORD *)a3 - 4) == *(_QWORD *)(a1 + 16);
      }
    }
  }
  return 0;
}
