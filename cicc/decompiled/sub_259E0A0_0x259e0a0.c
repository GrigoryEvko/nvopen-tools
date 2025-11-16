// Function: sub_259E0A0
// Address: 0x259e0a0
//
__int64 __fastcall sub_259E0A0(__int64 a1, __int64 a2, __m128i *a3, int a4, _BYTE *a5, char a6, __int64 *a7)
{
  unsigned int v10; // r15d
  __int64 v12; // rax
  int v14[13]; // [rsp+1Ch] [rbp-34h] BYREF

  *a5 = 0;
  if ( (unsigned int)*(unsigned __int8 *)sub_250D070(a3) - 12 <= 1
    || *(_BYTE *)sub_250D070(a3) == 13
    || (v14[0] = 41, v10 = sub_2516400(a1, a3, (__int64)v14, 1, a6, 41), (_BYTE)v10) )
  {
    *a5 = 1;
    return 1;
  }
  else if ( a2 )
  {
    v12 = sub_259DC40(a1, a3->m128i_i64[0], a3->m128i_i64[1], a2, a4, 0, 1);
    if ( a7 )
      *a7 = v12;
    if ( v12 )
    {
      v10 = *(unsigned __int8 *)(v12 + 97);
      if ( (_BYTE)v10 )
        *a5 = *(_BYTE *)(v12 + 96);
    }
  }
  return v10;
}
