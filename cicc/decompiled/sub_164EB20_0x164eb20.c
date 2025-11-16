// Function: sub_164EB20
// Address: 0x164eb20
//
__int64 __fastcall sub_164EB20(_QWORD *a1, unsigned __int64 a2)
{
  unsigned int v2; // r12d
  _QWORD **v3; // r14
  unsigned int v4; // r13d
  _QWORD *v5; // rax

  v2 = 0;
  if ( a2 && (a2 & (a2 - 1)) == 0 )
  {
    v3 = (_QWORD **)*a1;
    v4 = *(_DWORD *)(*a1 + 8LL);
    if ( v4 <= 0x40 )
    {
      v5 = *v3;
      goto LABEL_6;
    }
    if ( v4 - (unsigned int)sub_16A57B0(*a1) <= 0x40 )
    {
      v5 = (_QWORD *)**v3;
LABEL_6:
      LOBYTE(v2) = a2 >= (unsigned __int64)v5;
    }
  }
  return v2;
}
