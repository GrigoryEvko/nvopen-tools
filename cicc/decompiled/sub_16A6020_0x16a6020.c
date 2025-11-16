// Function: sub_16A6020
// Address: 0x16a6020
//
void __fastcall sub_16A6020(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r15
  unsigned int v3; // r14d
  unsigned int v4; // r12d
  _QWORD *v5; // rcx
  __int64 v6; // rax
  __int64 v7; // rax

  v2 = *(unsigned int *)(a1 + 8);
  v3 = *(_DWORD *)(a2 + 8);
  v4 = *(_DWORD *)(a1 + 8);
  if ( v3 > 0x40 )
  {
    if ( v3 - (unsigned int)sub_16A57B0(a2) > 0x40 )
      goto LABEL_3;
    v5 = **(_QWORD ***)a2;
    if ( v2 < (unsigned __int64)v5 )
      goto LABEL_3;
  }
  else
  {
    v5 = *(_QWORD **)a2;
    if ( v2 < *(_QWORD *)a2 )
    {
LABEL_3:
      if ( v4 > 0x40 )
      {
LABEL_4:
        sub_16A5E70(a1, v4);
        return;
      }
      v6 = (__int64)(*(_QWORD *)a1 << (64 - (unsigned __int8)v4)) >> (64 - (unsigned __int8)v4);
LABEL_12:
      v7 = v6 >> 63;
      goto LABEL_10;
    }
  }
  if ( v4 > 0x40 )
  {
    v4 = (unsigned int)v5;
    goto LABEL_4;
  }
  v6 = (__int64)(*(_QWORD *)a1 << (64 - (unsigned __int8)v4)) >> (64 - (unsigned __int8)v4);
  if ( v4 == (_DWORD)v5 )
    goto LABEL_12;
  v7 = v6 >> (char)v5;
LABEL_10:
  *(_QWORD *)a1 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v4) & v7;
}
