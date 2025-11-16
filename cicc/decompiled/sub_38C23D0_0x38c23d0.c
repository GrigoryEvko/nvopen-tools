// Function: sub_38C23D0
// Address: 0x38c23d0
//
__int64 __fastcall sub_38C23D0(__int64 a1, __int64 a2, _BYTE *a3, unsigned int a4)
{
  __int64 result; // rax
  int v5; // r10d
  unsigned __int8 v6; // r8
  __int64 v7; // r11
  _BYTE *v8; // r12
  int v9; // r10d
  __int64 *v10; // rsi
  __int64 v11; // r9
  __int64 v12; // rsi

  if ( a3 || (result = a2, a4 != -1) )
  {
    v5 = *(_DWORD *)(a2 + 168);
    v6 = *(_BYTE *)(a2 + 148);
    v7 = *(_QWORD *)(a2 + 160);
    v8 = *(_BYTE **)(a2 + 152);
    if ( a3 )
    {
      v9 = v5 | 0x1000;
      if ( (*a3 & 4) != 0 )
      {
        v10 = (__int64 *)*((_QWORD *)a3 - 1);
        v11 = *v10;
        v12 = (__int64)(v10 + 2);
      }
      else
      {
        v11 = 0;
        v12 = 0;
      }
      return sub_38C20E0(a1, v8, v7, v9, v6, 5u, v12, v11, a4, 0);
    }
    else
    {
      return sub_38C20E0(a1, v8, v7, v5, v6, 0, (__int64)byte_3F871B3, 0, a4, 0);
    }
  }
  return result;
}
