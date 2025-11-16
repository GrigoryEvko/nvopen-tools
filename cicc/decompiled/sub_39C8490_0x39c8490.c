// Function: sub_39C8490
// Address: 0x39c8490
//
__int64 __fastcall sub_39C8490(__int64 *a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rcx

  v3 = sub_39C8350((__int64)a1, *a2, a2[1]);
  v4 = a2[2];
  if ( v3 && (v5 = *(_QWORD *)(v3 + 16)) != 0 )
    return sub_39A3B20((__int64)a1, v4, 49, v5);
  else
    return sub_39C8370(a1, (__int64)a2, a2[2]);
}
