// Function: sub_86A320
// Address: 0x86a320
//
__int64 __fastcall sub_86A320(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v6; // rax
  __int64 result; // rax

  if ( dword_4F04C3C )
    return 0;
  v6 = sub_86A2A0(a1);
  if ( !v6 )
    return 0;
  result = *(_QWORD *)(v6 + 24);
  if ( a2 )
    *(_QWORD *)(result + 32) = a2;
  if ( a3 )
    *(_QWORD *)(result + 40) = a3;
  if ( (a4 & 1) != 0 )
    *(_BYTE *)(result + 57) |= 1u;
  if ( (a4 & 2) != 0 )
    *(_BYTE *)(result + 57) |= 4u;
  if ( (a4 & 4) != 0 )
    *(_BYTE *)(result + 57) |= 8u;
  if ( (a4 & 8) != 0 )
    *(_BYTE *)(result + 57) |= 0x10u;
  if ( (a4 & 0x10) != 0 )
    *(_BYTE *)(result + 57) |= 0x20u;
  if ( (a4 & 0x40) != 0 )
    *(_BYTE *)(result + 57) |= 0x40u;
  return result;
}
