// Function: sub_169B240
// Address: 0x169b240
//
char __fastcall sub_169B240(__int64 a1, __int64 *a2)
{
  __int64 v2; // rbx
  unsigned int v3; // r14d
  unsigned int v4; // r13d
  unsigned int v5; // ebx
  _QWORD *v6; // rax

  if ( *((_DWORD *)a2 + 2) > 0x40u )
    a2 = (__int64 *)*a2;
  v2 = *a2;
  sub_1698320((_QWORD *)a1, (__int64)&unk_42AE9F0);
  v3 = v2 & 0x3FF;
  v4 = ((unsigned int)v2 >> 10) & 0x1F;
  v5 = *(_BYTE *)(a1 + 18) & 0xF7 | (8 * (((unsigned int)v2 >> 15) & 1));
  LOBYTE(v6) = v3 | v4;
  *(_BYTE *)(a1 + 18) = v5;
  if ( v3 | v4 )
  {
    LOBYTE(v6) = v4 == 31;
    if ( v3 || v4 != 31 )
    {
      if ( v3 && v4 == 31 )
      {
        *(_BYTE *)(a1 + 18) = *(_BYTE *)(a1 + 18) & 0xF8 | 1;
        v6 = (_QWORD *)sub_1698470(a1);
        *v6 = v3;
      }
      else
      {
        *(_BYTE *)(a1 + 18) = *(_BYTE *)(a1 + 18) & 0xF8 | 2;
        *(_WORD *)(a1 + 16) = v4 - 15;
        *(_QWORD *)sub_1698470(a1) = v3;
        if ( v4 )
        {
          v6 = (_QWORD *)sub_1698470(a1);
          *v6 |= 0x400uLL;
        }
        else
        {
          *(_WORD *)(a1 + 16) = -14;
          LOBYTE(v6) = -14;
        }
      }
    }
    else
    {
      *(_BYTE *)(a1 + 18) = v5 & 0xF8;
    }
  }
  else
  {
    *(_BYTE *)(a1 + 18) = v5 & 0xF8 | 3;
  }
  return (char)v6;
}
