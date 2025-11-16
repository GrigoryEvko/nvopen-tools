// Function: sub_15C9800
// Address: 0x15c9800
//
void __fastcall sub_15C9800(__int64 a1, _BYTE *a2, __int64 a3, _BYTE *a4, __int64 a5)
{
  __int64 *v8; // rdi

  *(_QWORD *)a1 = a1 + 16;
  if ( a2 )
  {
    sub_15C7EA0((__int64 *)a1, a2, (__int64)&a2[a3]);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_BYTE *)(a1 + 16) = 0;
  }
  v8 = (__int64 *)(a1 + 32);
  *(_QWORD *)(a1 + 32) = a1 + 48;
  if ( a4 )
  {
    sub_15C7EA0(v8, a4, (__int64)&a4[a5]);
  }
  else
  {
    *(_QWORD *)(a1 + 40) = 0;
    *(_BYTE *)(a1 + 48) = 0;
  }
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
}
