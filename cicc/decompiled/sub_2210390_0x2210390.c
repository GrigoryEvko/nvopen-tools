// Function: sub_2210390
// Address: 0x2210390
//
__int64 __fastcall sub_2210390(
        __int64 a1,
        __int64 a2,
        unsigned int *a3,
        unsigned int *a4,
        unsigned int **a5,
        _BYTE *a6,
        __int64 a7,
        _QWORD *a8)
{
  unsigned int *v8; // r10
  unsigned int *v9; // r11
  unsigned int v10; // esi
  _BYTE *v11; // r9
  __int64 result; // rax
  _BYTE *v13; // r9
  _BYTE *v14; // r9
  _BYTE *v15; // [rsp+0h] [rbp-10h] BYREF
  __int64 v16; // [rsp+8h] [rbp-8h]

  v8 = a3;
  v15 = a6;
  v16 = a7;
  if ( a3 == a4 )
  {
    result = 0;
LABEL_11:
    *a5 = v8;
    *a8 = a6;
  }
  else
  {
    v9 = a4;
    do
    {
      v10 = *v8;
      if ( *v8 > 0x10FFFF )
      {
        v13 = v15;
        *a5 = v8;
        *a8 = v13;
        return 2;
      }
      if ( v10 <= 0x7F )
      {
        a6 = v15;
        if ( (_BYTE *)v16 == v15 )
        {
          result = 1;
          goto LABEL_11;
        }
        ++v15;
        *a6 = v10;
      }
      else if ( !(unsigned __int8)sub_220FEB0((__int64)&v15, v10) )
      {
        v11 = v15;
        *a5 = v8;
        *a8 = v11;
        return 1;
      }
      ++v8;
    }
    while ( v9 != v8 );
    v14 = v15;
    *a5 = v8;
    *a8 = v14;
    return 0;
  }
  return result;
}
