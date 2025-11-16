// Function: sub_23C6530
// Address: 0x23c6530
//
__int64 __fastcall sub_23C6530(__int64 a1, __int64 a2)
{
  char v3; // al
  __int64 v4; // rcx
  __int16 v5; // dx
  __int64 result; // rax

  *(_QWORD *)a1 = a1 + 16;
  sub_23C6360((__int64 *)a1, *(_BYTE **)a2, *(_QWORD *)a2 + *(_QWORD *)(a2 + 8));
  *(_QWORD *)(a1 + 32) = a1 + 48;
  sub_23C6360((__int64 *)(a1 + 32), *(_BYTE **)(a2 + 32), *(_QWORD *)(a2 + 32) + *(_QWORD *)(a2 + 40));
  *(_QWORD *)(a1 + 64) = a1 + 80;
  sub_23C6360((__int64 *)(a1 + 64), *(_BYTE **)(a2 + 64), *(_QWORD *)(a2 + 64) + *(_QWORD *)(a2 + 72));
  *(_QWORD *)(a1 + 96) = a1 + 112;
  sub_23C6360((__int64 *)(a1 + 96), *(_BYTE **)(a2 + 96), *(_QWORD *)(a2 + 96) + *(_QWORD *)(a2 + 104));
  v3 = *(_BYTE *)(a2 + 142);
  v4 = *(_QWORD *)(a2 + 128);
  *(_DWORD *)(a1 + 136) = *(_DWORD *)(a2 + 136);
  v5 = *(_WORD *)(a2 + 140);
  *(_QWORD *)(a1 + 128) = v4;
  *(_WORD *)(a1 + 140) = v5;
  *(_BYTE *)(a1 + 142) = v3;
  result = *(_QWORD *)(a2 + 144);
  *(_QWORD *)(a1 + 144) = result;
  if ( result )
    _InterlockedAdd((volatile signed __int32 *)(result + 8), 1u);
  return result;
}
