// Function: sub_27F1130
// Address: 0x27f1130
//
__int64 __fastcall sub_27F1130(
        unsigned __int8 *a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        _BYTE *a7,
        __int64 a8)
{
  int v8; // eax
  unsigned __int64 v10; // rax
  void *v12; // r12
  __int64 v13; // rcx

  v8 = *a1;
  if ( (unsigned __int8)(v8 - 61) <= 0x18u )
  {
    v12 = &loc_100000B;
    if ( _bittest64((const __int64 *)&v12, (unsigned int)(v8 - 61)) )
      return sub_27F0130(a1, a2, a3, a4, a5, a6, a7, a8);
    v10 = (unsigned int)(v8 - 41);
  }
  else
  {
    v10 = (unsigned int)(v8 - 41);
    if ( (unsigned __int8)v10 > 0x37u )
      return 0;
  }
  v13 = 0xBE267FFC47FFFFLL;
  if ( !_bittest64(&v13, v10) )
    return 0;
  return sub_27F0130(a1, a2, a3, a4, a5, a6, a7, a8);
}
