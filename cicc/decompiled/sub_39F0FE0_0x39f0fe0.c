// Function: sub_39F0FE0
// Address: 0x39f0fe0
//
void __fastcall sub_39F0FE0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax

  sub_38D6650(a1, a2, a3, a4);
  v4 = *(unsigned int *)(a1 + 120);
  if ( !(_DWORD)v4 )
    BUG();
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v4 - 32) + 173LL) & 4) != 0 )
    sub_38E28A0((__int64)a2, 6);
}
