// Function: sub_16A88B0
// Address: 0x16a88b0
//
__int64 __fastcall sub_16A88B0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v4; // eax
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax

  v4 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 8) = v4;
  if ( v4 <= 0x40 )
  {
    *(_QWORD *)a1 = 0;
    if ( !a3 )
    {
      v7 = 0;
      goto LABEL_6;
    }
    if ( a3 <= 0x40 )
    {
      v5 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)a3);
      v6 = 0;
LABEL_5:
      v7 = v5 | v6;
      *(_QWORD *)a1 = v7;
LABEL_6:
      *(_QWORD *)a1 = *(_QWORD *)a2 & v7;
      return a1;
    }
LABEL_14:
    sub_16A5260((_QWORD *)a1, 0, a3);
    if ( *(_DWORD *)(a1 + 8) <= 0x40u )
      goto LABEL_13;
    goto LABEL_12;
  }
  sub_16A4EF0(a1, 0, 0);
  if ( a3 )
  {
    if ( a3 > 0x40 )
      goto LABEL_14;
    v5 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)a3);
    v6 = *(_QWORD *)a1;
    if ( *(_DWORD *)(a1 + 8) <= 0x40u )
      goto LABEL_5;
    *(_QWORD *)v6 |= v5;
  }
  if ( *(_DWORD *)(a1 + 8) <= 0x40u )
  {
LABEL_13:
    v7 = *(_QWORD *)a1;
    goto LABEL_6;
  }
LABEL_12:
  sub_16A8890((__int64 *)a1, (__int64 *)a2);
  return a1;
}
