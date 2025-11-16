// Function: sub_34CEEA0
// Address: 0x34ceea0
//
char __fastcall sub_34CEEA0(__int64 a1, unsigned int a2, __int64 *a3)
{
  unsigned __int16 v3; // ax

  v3 = sub_2D5BAE0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16), a3, 0);
  if ( a2 > 4 )
    BUG();
  if ( v3 )
    LOBYTE(v3) = (((int)*(unsigned __int16 *)(*(_QWORD *)(a1 + 32) + 2 * (a2 + 5LL * v3 + 259392) + 10) >> 4) & 0xB) == 0;
  return v3;
}
