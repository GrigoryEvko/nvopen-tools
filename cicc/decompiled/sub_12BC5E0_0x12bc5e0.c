// Function: sub_12BC5E0
// Address: 0x12bc5e0
//
__int64 __fastcall sub_12BC5E0(__int64 *a1, __int64 *a2)
{
  __int64 v3; // rsi
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 result; // rax

  v3 = a1[1];
  if ( v3 == a1[2] )
    return sub_12BC360(a1, (char *)v3, a2);
  if ( v3 )
  {
    v4 = *a2;
    *a2 = 0;
    *(_QWORD *)v3 = v4;
    v5 = a2[1];
    a2[1] = 0;
    *(_QWORD *)(v3 + 8) = v5;
    v6 = a2[2];
    a2[2] = 0;
    *(_QWORD *)(v3 + 16) = v6;
    result = a2[3];
    a2[3] = 0;
    *(_QWORD *)(v3 + 24) = result;
    v3 = a1[1];
  }
  a1[1] = v3 + 32;
  return result;
}
