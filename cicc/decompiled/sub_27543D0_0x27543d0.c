// Function: sub_27543D0
// Address: 0x27543d0
//
bool __fastcall sub_27543D0(unsigned __int8 *a1)
{
  int v1; // eax
  bool result; // al
  __int64 v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rdi
  unsigned int v6; // ebx

  v1 = *a1;
  if ( (_BYTE)v1 == 62 )
  {
    result = !(*((_WORD *)a1 + 1) & 1);
    if ( ((*((_WORD *)a1 + 1) >> 7) & 6) == 0 )
      return result;
    return 0;
  }
  if ( (unsigned __int8)(v1 - 34) > 0x33u )
    return 0;
  v3 = 0x8000000000041LL;
  if ( !_bittest64(&v3, (unsigned int)(v1 - 34)) )
    return 0;
  if ( (_BYTE)v1 != 85
    || (v4 = *((_QWORD *)a1 - 4)) == 0
    || *(_BYTE *)v4
    || *(_QWORD *)(v4 + 24) != *((_QWORD *)a1 + 10)
    || (*(_BYTE *)(v4 + 33) & 0x20) == 0
    || (unsigned int)(*(_DWORD *)(v4 + 36) - 238) > 7
    || ((1LL << (*(_BYTE *)(v4 + 36) + 18)) & 0xAD) == 0 )
  {
    if ( !sub_B46A10((__int64)a1)
      && !*((_QWORD *)a1 + 2)
      && (unsigned __int8)sub_B46900(a1)
      && ((unsigned __int8)sub_A73ED0((_QWORD *)a1 + 9, 41) || (unsigned __int8)sub_B49560((__int64)a1, 41)) )
    {
      return (unsigned int)*a1 - 30 > 0xA;
    }
    return 0;
  }
  v5 = *(_QWORD *)&a1[32 * (3LL - (*((_DWORD *)a1 + 1) & 0x7FFFFFF))];
  v6 = *(_DWORD *)(v5 + 32);
  if ( v6 <= 0x40 )
    return *(_QWORD *)(v5 + 24) == 0;
  else
    return v6 == (unsigned int)sub_C444A0(v5 + 24);
}
