// Function: sub_2B17600
// Address: 0x2b17600
//
__int64 __fastcall sub_2B17600(_BYTE **a1, __int64 a2)
{
  __int64 v2; // rsi
  unsigned __int8 **v3; // rax
  _QWORD *v4; // r8
  unsigned __int8 *v5; // r13
  unsigned __int8 **v6; // rbx
  _QWORD *v7; // r8
  __int64 result; // rax
  unsigned __int8 **v9; // rax
  unsigned __int8 v10; // dl

  v2 = (__int64)&a1[a2];
  v3 = sub_2B0CB90(a1, v2);
  if ( (unsigned __int8 **)v2 == v3 )
    return 0;
  v5 = *v3;
  v6 = v3;
  v7 = sub_2B0BF30(v4, v2, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B15E10);
  result = 1;
  if ( (_QWORD *)v2 != v7 )
  {
    v9 = v6;
    while ( 1 )
    {
      v10 = **v9;
      if ( v10 != 13 && (v10 <= 0x1Cu || *((_QWORD *)v5 + 5) != *((_QWORD *)*v9 + 5)) )
        break;
      if ( (unsigned __int8 **)v2 == ++v9 )
        return 1;
    }
    return 0;
  }
  return result;
}
