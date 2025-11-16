// Function: sub_AD24A0
// Address: 0xad24a0
//
__int64 __fastcall sub_AD24A0(__int64 **a1, __int64 *a2, __int64 a3)
{
  _BYTE *v3; // rdi
  int v4; // eax
  char v5; // bl
  bool v6; // r13
  char v7; // bl
  __int64 *v8; // r12
  unsigned __int8 *v9; // r14
  bool v10; // zf
  int v11; // eax
  bool v15; // [rsp+27h] [rbp-39h]
  __int64 *v16; // [rsp+28h] [rbp-38h]

  if ( !a3 )
    return sub_AC9350(a1);
  v3 = (_BYTE *)*a2;
  v4 = *(unsigned __int8 *)*a2;
  v5 = *(_BYTE *)*a2;
  v15 = (_BYTE)v4 == 13;
  if ( (unsigned int)(v4 - 12) <= 1 )
  {
    v6 = sub_AC30F0((__int64)v3);
    v16 = &a2[a3];
    if ( a2 == v16 )
    {
      if ( !v6 )
      {
        if ( v5 != 13 )
          return sub_ACA8A0(a1);
        return sub_ACADE0(a1);
      }
      return sub_AC9350(a1);
    }
    v7 = 1;
LABEL_5:
    v8 = a2;
    do
    {
      while ( 1 )
      {
        v9 = (unsigned __int8 *)*v8;
        v10 = !sub_AC30F0(*v8);
        v11 = *v9;
        if ( v10 )
          v6 = 0;
        if ( (_BYTE)v11 != 13 )
          break;
        v7 = 0;
        if ( v16 == ++v8 )
          goto LABEL_13;
      }
      v15 = 0;
      if ( (unsigned int)(v11 - 12) >= 2 )
        v7 = 0;
      ++v8;
    }
    while ( v16 != v8 );
LABEL_13:
    if ( !v6 )
    {
      if ( !v15 )
      {
        if ( v7 )
          return sub_ACA8A0(a1);
        return sub_AD22F0(**a1 + 1776, (__int64)a1, a2, a3);
      }
      return sub_ACADE0(a1);
    }
    return sub_AC9350(a1);
  }
  v6 = sub_AC30F0((__int64)v3);
  if ( v6 )
  {
    v16 = &a2[a3];
    if ( a2 == v16 )
      return sub_AC9350(a1);
    v7 = 0;
    goto LABEL_5;
  }
  if ( v5 != 13 )
    return sub_AD22F0(**a1 + 1776, (__int64)a1, a2, a3);
  return sub_ACADE0(a1);
}
