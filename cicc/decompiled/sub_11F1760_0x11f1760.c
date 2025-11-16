// Function: sub_11F1760
// Address: 0x11f1760
//
__int64 __fastcall sub_11F1760(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned int v5; // eax

  v4 = sub_B43CA0(a2);
  v5 = 8 * (unsigned int)sub_97FA40(**(_QWORD **)(a1 + 24), v4);
  if ( v5 )
    return sub_11F0480(a1, a2, a3, v5, 0);
  else
    return 0;
}
