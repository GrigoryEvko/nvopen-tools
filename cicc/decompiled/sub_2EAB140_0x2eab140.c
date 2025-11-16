// Function: sub_2EAB140
// Address: 0x2eab140
//
void __fastcall sub_2EAB140(unsigned int *a1, int a2, unsigned int a3, __int64 a4)
{
  int v4; // r12d

  if ( a3 )
  {
    LOWORD(v4) = a3;
    if ( ((*a1 >> 8) & 0xFFF) == 0 )
    {
      sub_2EAB0C0((__int64)a1, a2);
LABEL_4:
      *a1 = *a1 & 0xFFF000FF | ((v4 & 0xFFF) << 8);
      return;
    }
    v4 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a4 + 296LL))(a4, a3);
    sub_2EAB0C0((__int64)a1, a2);
    if ( v4 )
      goto LABEL_4;
  }
  else
  {
    sub_2EAB0C0((__int64)a1, a2);
  }
}
