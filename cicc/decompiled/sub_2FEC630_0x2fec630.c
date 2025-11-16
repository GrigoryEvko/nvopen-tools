// Function: sub_2FEC630
// Address: 0x2fec630
//
__int64 __fastcall sub_2FEC630(__int64 a1, _BYTE *a2)
{
  unsigned int (*v2)(void); // rax
  unsigned int v3; // r12d

  if ( (unsigned __int8)(*a2 - 65) > 1u )
    BUG();
  v2 = *(unsigned int (**)(void))(*(_QWORD *)a1 + 88LL);
  v3 = (a2[2] & 1) == 0 ? 3 : 7;
  if ( (char *)v2 == (char *)sub_2FE2E30 )
    return v3;
  else
    return v2() | v3;
}
