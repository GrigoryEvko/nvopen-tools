// Function: sub_2FEC2E0
// Address: 0x2fec2e0
//
__int64 __fastcall sub_2FEC2E0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 *a6)
{
  unsigned __int8 v8; // al
  unsigned int v9; // r12d
  __int64 v10; // rbx
  unsigned int v11; // eax

  v8 = sub_11F3070(*(_QWORD *)(a2 + 40), a5, a6);
  v9 = v8;
  v10 = (unsigned int)sub_2FEC250(a1, v8);
  v11 = sub_2FEC2D0(a1);
  if ( (_BYTE)v9 || v11 >= a4 )
    LOBYTE(v9) = 100 * a3 >= a4 * v10;
  return v9;
}
