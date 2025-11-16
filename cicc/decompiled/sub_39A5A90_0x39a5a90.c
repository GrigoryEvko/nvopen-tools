// Function: sub_39A5A90
// Address: 0x39a5a90
//
__int64 __fastcall sub_39A5A90(__int64 a1, __int16 a2, __int64 a3, unsigned __int8 *a4)
{
  __int64 v6; // r12
  unsigned __int64 *v7; // rax

  v6 = sub_145CBF0((__int64 *)(a1 + 88), 48, 16);
  *(_QWORD *)v6 = v6 | 4;
  v7 = *(unsigned __int64 **)(a3 + 32);
  *(_QWORD *)(v6 + 8) = 0;
  *(_QWORD *)(v6 + 16) = 0;
  *(_DWORD *)(v6 + 24) = -1;
  *(_WORD *)(v6 + 28) = a2;
  *(_BYTE *)(v6 + 30) = 0;
  *(_QWORD *)(v6 + 32) = 0;
  *(_QWORD *)(v6 + 40) = a3;
  if ( v7 )
  {
    *(_QWORD *)v6 = *v7;
    *v7 = v6 & 0xFFFFFFFFFFFFFFFBLL;
  }
  *(_QWORD *)(a3 + 32) = v6;
  if ( a4 )
    sub_39A55B0(a1, a4, (unsigned __int8 *)v6);
  return v6;
}
