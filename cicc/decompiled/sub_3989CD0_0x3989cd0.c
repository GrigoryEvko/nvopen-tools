// Function: sub_3989CD0
// Address: 0x3989cd0
//
__int64 __fastcall sub_3989CD0(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v4; // rsi
  __int64 result; // rax
  __int64 v7; // rbx
  _QWORD *v8; // rdx

  v4 = *(_BYTE **)(a3 - 8LL * *(unsigned int *)(a3 + 8));
  result = (unsigned int)(unsigned __int8)*v4 - 17;
  if ( (unsigned __int8)(*v4 - 17) > 2u )
  {
    result = sub_39A81B0(a2);
    v7 = result;
    if ( result )
    {
      result = sub_39CC1A0(a2, a3);
      *(_QWORD *)(result + 40) = v7;
      v8 = *(_QWORD **)(v7 + 32);
      if ( v8 )
      {
        *(_QWORD *)result = *v8;
        **(_QWORD **)(v7 + 32) = result & 0xFFFFFFFFFFFFFFFBLL;
      }
      *(_QWORD *)(v7 + 32) = result;
    }
  }
  return result;
}
