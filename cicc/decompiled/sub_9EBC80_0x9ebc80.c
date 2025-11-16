// Function: sub_9EBC80
// Address: 0x9ebc80
//
__int64 __fastcall sub_9EBC80(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r12
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 result; // rax

  v2 = *(_QWORD **)(a1 + 8);
  if ( v2 == *(_QWORD **)(a1 + 16) )
    return sub_9EB9D0(a1, *(char **)(a1 + 8), a2);
  if ( v2 )
  {
    v2[1] = 0;
    *v2 = v2 + 3;
    v2[2] = 40;
    if ( *(_QWORD *)(a2 + 8) )
      sub_9C3060((__int64)v2, (char **)a2);
    v3 = *(_QWORD *)(a2 + 64);
    *(_QWORD *)(a2 + 64) = 0;
    v2[8] = v3;
    v4 = *(_QWORD *)(a2 + 72);
    *(_QWORD *)(a2 + 72) = 0;
    v2[9] = v4;
    v5 = *(_QWORD *)(a2 + 80);
    *(_QWORD *)(a2 + 80) = 0;
    v2[10] = v5;
    v6 = *(_QWORD *)(a2 + 88);
    *(_QWORD *)(a2 + 88) = 0;
    v2[11] = v6;
    v7 = *(_QWORD *)(a2 + 96);
    *(_QWORD *)(a2 + 96) = 0;
    v2[12] = v7;
    result = *(_QWORD *)(a2 + 104);
    *(_QWORD *)(a2 + 104) = 0;
    v2[13] = result;
    v2 = *(_QWORD **)(a1 + 8);
  }
  *(_QWORD *)(a1 + 8) = v2 + 14;
  return result;
}
