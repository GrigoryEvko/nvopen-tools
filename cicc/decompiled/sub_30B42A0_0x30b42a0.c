// Function: sub_30B42A0
// Address: 0x30b42a0
//
__int64 *__fastcall sub_30B42A0(__int64 *a1, _BYTE *a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdx

  v8 = *a4;
  if ( *a2 )
    sub_30B3900(a1, a3, v8);
  else
    sub_30B4090(a1, a3, v8, a6);
  return a1;
}
