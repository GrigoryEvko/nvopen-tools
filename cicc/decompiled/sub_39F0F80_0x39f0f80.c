// Function: sub_39F0F80
// Address: 0x39f0f80
//
void __fastcall sub_39F0F80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax

  sub_38D6580(a1, a2, a3);
  v3 = *(unsigned int *)(a1 + 120);
  if ( !(_DWORD)v3 )
    BUG();
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v3 - 32) + 173LL) & 4) != 0 )
    sub_38E28A0(a2, 6);
}
