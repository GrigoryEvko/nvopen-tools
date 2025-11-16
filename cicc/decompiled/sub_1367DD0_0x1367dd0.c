// Function: sub_1367DD0
// Address: 0x1367dd0
//
__int64 __fastcall sub_1367DD0(__int64 a1, __int64 a2, unsigned int *a3)
{
  __int64 v3; // rdx
  _BYTE *v4; // rsi

  v4 = (_BYTE *)sub_1649960(*(_QWORD *)(*(_QWORD *)(a2 + 136) + 8LL * *a3));
  *(_QWORD *)a1 = a1 + 16;
  if ( v4 )
  {
    sub_1367D20((__int64 *)a1, v4, (__int64)&v4[v3]);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_BYTE *)(a1 + 16) = 0;
  }
  return a1;
}
