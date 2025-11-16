// Function: sub_85E1C0
// Address: 0x85e1c0
//
__int64 __fastcall sub_85E1C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, unsigned int a7)
{
  int v8; // eax
  int v9; // esi
  int v10; // ebx
  __int64 result; // rax
  _DWORD *v12; // rdx

  v8 = 0;
  if ( a5 && (*(_BYTE *)(a5 + 81) & 0x10) != 0 )
    v8 = *(_BYTE *)(*(_QWORD *)(a5 + 64) + 89LL) & 1;
  v9 = *(_DWORD *)(a1 + 8);
  dword_4F04C38 = v8;
  v10 = unk_4F04C2C;
  sub_85C120(9u, v9, a2, a3, 0, a4, a5, a6, a1, 0, 0, 0, a7);
  if ( dword_4F04C64 == -1 )
  {
    MEMORY[0x23C] = 0;
    BUG();
  }
  result = (unsigned int)(dword_4F04C64 - 1);
  v12 = (_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
  v12[143] = result;
  v12[144] = v10;
  v12[139] = result;
  return result;
}
