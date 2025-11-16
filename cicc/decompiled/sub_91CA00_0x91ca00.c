// Function: sub_91CA00
// Address: 0x91ca00
//
char __fastcall sub_91CA00(__int64 a1)
{
  const char *v1; // rax
  _BOOL4 v2; // r13d
  _QWORD *v3; // rax
  _QWORD *v4; // r12
  char v5; // al
  char v6; // al

  LOBYTE(v1) = sub_91C2E0(a1);
  if ( (_BYTE)v1 )
  {
    v2 = sub_8D3030(a1);
    v3 = sub_7259C0(2);
    *((_BYTE *)v3 + 160) = 10;
    v4 = v3;
    v3[16] = 0;
    sub_8D6090((__int64)v3);
    sub_725570(a1, 12);
    v5 = *(_BYTE *)(a1 - 8);
    *(_QWORD *)(a1 + 160) = v4;
    v6 = v5 | 0x20;
    *(_BYTE *)(a1 - 8) = v6;
    if ( v2 )
    {
      *(_QWORD *)(a1 + 8) = "__surface_type__";
      LOBYTE(v1) = (v2 << 6) | v6 & 0xBF;
      *(_BYTE *)(a1 - 8) = (_BYTE)v1;
    }
    else
    {
      v1 = "__texture_type__";
      *(_QWORD *)(a1 + 8) = "__texture_type__";
    }
  }
  return (char)v1;
}
