// Function: sub_DF9740
// Address: 0xdf9740
//
__int64 __fastcall sub_DF9740(__int64 *a1, unsigned __int8 *a2)
{
  int v2; // eax
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rdi
  __int64 (*v7)(); // rax

  v2 = *a2;
  if ( (unsigned __int8)v2 > 0x1Cu )
  {
    v4 = (unsigned int)(v2 - 34);
    if ( (unsigned __int8)v4 <= 0x33u )
    {
      v5 = 0x8000000000041LL;
      if ( _bittest64(&v5, v4) )
      {
        if ( (unsigned __int8)sub_A73ED0((_QWORD *)a2 + 9, 26) || (unsigned __int8)sub_B49560((__int64)a2, 26) )
          return 0;
      }
    }
  }
  v6 = *a1;
  v7 = *(__int64 (**)())(*(_QWORD *)*a1 + 152LL);
  if ( v7 == sub_DF5C30 )
    return 0;
  else
    return ((__int64 (__fastcall *)(__int64, unsigned __int8 *))v7)(v6, a2);
}
