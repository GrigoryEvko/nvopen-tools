// Function: sub_D95050
// Address: 0xd95050
//
__int64 __fastcall sub_D95050(__int64 a1, char a2, char a3)
{
  int v5; // edx
  int v6; // esi
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 *v9; // r12
  __int64 result; // rax

  *(_QWORD *)a1 = &unk_49DC150;
  v5 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  v6 = *(_DWORD *)(a1 + 12) & 0x8000;
  *(_WORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 80) = 0x100000000LL;
  *(_DWORD *)(a1 + 8) = v5;
  *(_QWORD *)(a1 + 24) = 0;
  *(_DWORD *)(a1 + 12) = v6 | a2 & 7 | (32 * a3) & 0x60;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = a1 + 128;
  *(_QWORD *)(a1 + 112) = 1;
  *(_DWORD *)(a1 + 120) = 0;
  *(_BYTE *)(a1 + 124) = 1;
  v9 = sub_C57470();
  result = *(unsigned int *)(a1 + 80);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), result + 1, 8u, v7, v8);
    result = *(unsigned int *)(a1 + 80);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * result) = v9;
  ++*(_DWORD *)(a1 + 80);
  return result;
}
