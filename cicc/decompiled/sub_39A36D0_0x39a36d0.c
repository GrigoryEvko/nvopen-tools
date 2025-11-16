// Function: sub_39A36D0
// Address: 0x39a36d0
//
void __fastcall sub_39A36D0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  unsigned int v5; // eax
  _BYTE v6[2]; // [rsp-2Ch] [rbp-2Ch] BYREF
  char v7; // [rsp-2Ah] [rbp-2Ah]

  if ( a3 )
  {
    v5 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 48LL))(a1, a4);
    v7 = 0;
    sub_39A3560(a1, (__int64 *)(a2 + 8), 58, (__int64)v6, v5);
    v7 = 0;
    sub_39A3560(a1, (__int64 *)(a2 + 8), 59, (__int64)v6, a3);
  }
}
