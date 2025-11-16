// Function: sub_DC3B80
// Address: 0xdc3b80
//
__int64 __fastcall sub_DC3B80(__int64 a1, int a2, _BYTE *a3, _BYTE *a4)
{
  unsigned int v4; // r12d
  __int64 v8; // rax
  _BYTE *v9; // rax

  v4 = 0;
  if ( a2 == 36 )
  {
    v4 = *(unsigned __int8 *)(a1 + 609);
    if ( (_BYTE)v4 )
    {
      return 0;
    }
    else
    {
      *(_BYTE *)(a1 + 609) = 1;
      if ( (unsigned __int8)sub_DBED40(a1, (__int64)a4) )
      {
        v8 = sub_D95540((__int64)a3);
        v9 = sub_DA2C50(a1, v8, 0, 0);
        if ( (unsigned __int8)sub_DC3A60(a1, 39, a3, v9) )
          v4 = sub_DC3A60(a1, 40, a3, a4);
      }
      *(_BYTE *)(a1 + 609) = 0;
    }
  }
  return v4;
}
