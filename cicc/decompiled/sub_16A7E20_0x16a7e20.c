// Function: sub_16A7E20
// Address: 0x16a7e20
//
__int64 __fastcall sub_16A7E20(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r15
  unsigned int v3; // r14d
  unsigned int v4; // r13d
  _QWORD *v5; // rax
  unsigned __int64 v7; // rax

  v2 = *(unsigned int *)(a1 + 8);
  v3 = *(_DWORD *)(a2 + 8);
  v4 = *(_DWORD *)(a1 + 8);
  if ( v3 <= 0x40 )
  {
    v5 = *(_QWORD **)a2;
    if ( v2 < *(_QWORD *)a2 )
      goto LABEL_3;
LABEL_10:
    if ( v4 > 0x40 )
    {
      v4 = (unsigned int)v5;
      goto LABEL_4;
    }
    if ( v4 != (_DWORD)v5 )
    {
      v7 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v4) & (*(_QWORD *)a1 << (char)v5);
      goto LABEL_7;
    }
LABEL_6:
    v7 = 0;
LABEL_7:
    *(_QWORD *)a1 = v7;
    return a1;
  }
  if ( v3 - (unsigned int)sub_16A57B0(a2) <= 0x40 )
  {
    v5 = **(_QWORD ***)a2;
    if ( v2 >= (unsigned __int64)v5 )
      goto LABEL_10;
  }
LABEL_3:
  if ( v4 <= 0x40 )
    goto LABEL_6;
LABEL_4:
  sub_16A7DC0((__int64 *)a1, v4);
  return a1;
}
