// Function: sub_1DD5940
// Address: 0x1dd5940
//
__int64 __fastcall sub_1DD5940(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  bool v4; // zf
  __int64 v5; // [rsp+0h] [rbp-20h] BYREF
  char v6; // [rsp+8h] [rbp-18h]

  result = a1 + 24;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 24) = (a1 + 24) | 4;
  *(_QWORD *)(a1 + 32) = a1 + 24;
  *(_QWORD *)(a1 + 40) = a3;
  *(_DWORD *)(a1 + 48) = -1;
  *(_QWORD *)(a1 + 56) = a2;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_BYTE *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_BYTE *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 16) = a1;
  if ( a3 )
  {
    result = sub_157F7D0((__int64)&v5, a3);
    if ( v6 )
    {
      result = v5;
      v4 = *(_BYTE *)(a1 + 144) == 0;
      *(_QWORD *)(a1 + 136) = v5;
      if ( v4 )
        *(_BYTE *)(a1 + 144) = 1;
    }
    else if ( *(_BYTE *)(a1 + 144) )
    {
      *(_BYTE *)(a1 + 144) = 0;
    }
  }
  return result;
}
