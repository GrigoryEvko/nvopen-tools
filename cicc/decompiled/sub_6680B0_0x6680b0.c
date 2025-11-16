// Function: sub_6680B0
// Address: 0x6680b0
//
__int64 __fastcall sub_6680B0(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // rdi

  if ( (unsigned int)sub_8D2310(a1) || (unsigned int)sub_8D3410(a1) )
  {
    v4 = 2777;
LABEL_4:
    sub_685360(v4, a2);
    return sub_72C930(v4);
  }
  if ( !a3 && (*(_BYTE *)(a1 + 140) & 0xFB) == 8 && (unsigned int)sub_8D4C10(a1, dword_4F077C4 != 2) )
  {
    v4 = 2778;
    goto LABEL_4;
  }
  return sub_73C570(a1, 8, -1);
}
