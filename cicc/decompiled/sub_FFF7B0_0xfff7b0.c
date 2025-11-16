// Function: sub_FFF7B0
// Address: 0xfff7b0
//
__int64 __fastcall sub_FFF7B0(__int64 a1, unsigned __int8 *a2, _BYTE *a3)
{
  int v4; // eax
  __int64 v5; // rsi
  int v6; // eax
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // [rsp+0h] [rbp-30h] BYREF
  int v11; // [rsp+8h] [rbp-28h]
  __int64 v12; // [rsp+10h] [rbp-20h]
  int v13; // [rsp+18h] [rbp-18h]

  v4 = *a2;
  if ( (unsigned __int8)v4 > 0x1Cu )
  {
    if ( !*a3 || (a2[7] & 0x20) == 0 )
      goto LABEL_12;
    v5 = sub_B91C10((__int64)a2, 4);
    if ( v5 )
    {
      sub_ABEA30((__int64)&v10, v5);
      v6 = v11;
      *(_BYTE *)(a1 + 32) = 1;
      *(_DWORD *)(a1 + 8) = v6;
      *(_QWORD *)a1 = v10;
      *(_DWORD *)(a1 + 24) = v13;
      *(_QWORD *)(a1 + 16) = v12;
      return a1;
    }
    v4 = *a2;
    if ( (_BYTE)v4 != 22 )
    {
      if ( (unsigned __int8)v4 <= 0x1Cu )
        goto LABEL_11;
LABEL_12:
      v8 = (unsigned int)(v4 - 34);
      if ( (unsigned __int8)v8 <= 0x33u )
      {
        v9 = 0x8000000000041LL;
        if ( _bittest64(&v9, v8) )
        {
          sub_B492D0(a1, (__int64)a2);
          return a1;
        }
      }
      goto LABEL_11;
    }
LABEL_8:
    sub_B2D8F0(a1, (__int64)a2);
    return a1;
  }
  if ( (_BYTE)v4 == 22 )
    goto LABEL_8;
LABEL_11:
  *(_BYTE *)(a1 + 32) = 0;
  return a1;
}
