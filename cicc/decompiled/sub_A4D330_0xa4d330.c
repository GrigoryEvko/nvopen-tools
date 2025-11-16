// Function: sub_A4D330
// Address: 0xa4d330
//
__int64 __fastcall sub_A4D330(__int64 a1, __int64 *a2)
{
  __int64 v3; // rsi
  __int64 v4; // rax
  __int64 result; // rax

  v3 = *(_QWORD *)(a1 + 8);
  if ( v3 == *(_QWORD *)(a1 + 16) )
    return sub_A1A390((char **)a1, (char *)v3, a2);
  if ( v3 )
  {
    v4 = *a2;
    *(_QWORD *)(v3 + 8) = 0;
    *a2 = 0;
    *(_QWORD *)v3 = v4;
    result = a2[1];
    a2[1] = 0;
    *(_QWORD *)(v3 + 8) = result;
    v3 = *(_QWORD *)(a1 + 8);
  }
  *(_QWORD *)(a1 + 8) = v3 + 16;
  return result;
}
