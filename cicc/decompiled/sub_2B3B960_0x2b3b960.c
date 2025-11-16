// Function: sub_2B3B960
// Address: 0x2b3b960
//
bool __fastcall sub_2B3B960(_QWORD **a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v5; // rdx
  _QWORD *v6; // rsi
  __int64 i; // rax
  _BYTE *v9; // r9

  v5 = (_QWORD *)**a1;
  v6 = &v5[*((unsigned int *)*a1 + 2)];
  if ( v5 != v6 )
  {
    for ( i = 0; i != *(_DWORD *)(a2 + 8); ++i )
    {
      if ( *(_DWORD *)(*a3 + 4 * i) != -1 )
      {
        v9 = *(_BYTE **)(*(_QWORD *)a2 + 8 * i);
        if ( (unsigned __int8)(*v9 - 12) > 1u && v9 != (_BYTE *)*v5 )
          break;
      }
      if ( v6 == ++v5 )
        return 1;
    }
  }
  return v6 == v5;
}
