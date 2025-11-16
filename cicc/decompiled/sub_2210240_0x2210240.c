// Function: sub_2210240
// Address: 0x2210240
//
__int64 __fastcall sub_2210240(
        __int64 a1,
        __int64 a2,
        unsigned __int16 *a3,
        unsigned __int16 *a4,
        unsigned __int16 **a5,
        _BYTE *a6,
        __int64 a7,
        _QWORD *a8)
{
  unsigned __int16 *v8; // r11
  unsigned __int16 *v9; // r10
  unsigned __int64 v10; // rcx
  int v11; // eax
  unsigned int v12; // esi
  int v13; // r9d
  unsigned __int16 *v14; // r10
  unsigned __int16 v15; // ax
  _BYTE *v16; // r9
  __int64 result; // rax
  _BYTE *v18; // r9
  _BYTE *v19; // r9
  _BYTE *v20; // [rsp+0h] [rbp-10h] BYREF
  __int64 v21; // [rsp+8h] [rbp-8h]

  v8 = a4;
  v9 = a3;
  v10 = (char *)a4 - (char *)a3;
  v20 = a6;
  v21 = a7;
  if ( a3 != v8 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v12 = *v9;
        v15 = *v9;
        if ( v12 - 55296 > 0x3FF )
          break;
        if ( v10 <= 2 )
          goto LABEL_13;
        v11 = v9[1];
        if ( (unsigned int)(v11 - 56320) > 0x3FF )
        {
LABEL_16:
          v18 = v20;
          *a5 = v9;
          *a8 = v18;
          return 2;
        }
        v12 = v11 + (v12 << 10) - 56613888;
LABEL_6:
        if ( !(unsigned __int8)sub_220FEB0((__int64)&v20, v12) )
        {
          v19 = v20;
          *a5 = v14;
          *a8 = v19;
          return 1;
        }
        v9 = &v14[v13];
        v10 = (char *)v8 - (char *)v9;
        if ( v8 == v9 )
          goto LABEL_13;
      }
      if ( v12 - 56320 <= 0x3FF )
        goto LABEL_16;
      if ( v12 > 0x7F )
        goto LABEL_6;
      a6 = v20;
      if ( (_BYTE *)v21 == v20 )
      {
        result = 1;
        goto LABEL_15;
      }
      ++v20;
      *a6 = v15;
      v10 = (char *)v8 - (char *)++v9;
      if ( v8 == v9 )
      {
LABEL_13:
        v16 = v20;
        *a5 = v9;
        *a8 = v16;
        return 0;
      }
    }
  }
  result = 0;
LABEL_15:
  *a5 = v9;
  *a8 = a6;
  return result;
}
