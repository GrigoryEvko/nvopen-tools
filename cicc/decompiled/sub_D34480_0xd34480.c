// Function: sub_D34480
// Address: 0xd34480
//
__int64 __fastcall sub_D34480(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 result; // rax

  v3 = 72LL * a2;
  v4 = v3 + *(_QWORD *)(a3 + 8);
  *(_QWORD *)a1 = *(_QWORD *)(v4 + 32);
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(v4 + 24);
  *(_QWORD *)(a1 + 16) = a1 + 32;
  *(_QWORD *)(a1 + 24) = 0x200000000LL;
  v5 = *(_QWORD *)(a3 + 8) + v3;
  v6 = *(_QWORD *)(*(_QWORD *)(v5 + 16) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 <= 1 )
    v6 = **(_QWORD **)(v6 + 16);
  *(_DWORD *)(a1 + 40) = *(_DWORD *)(v6 + 8) >> 8;
  result = *(unsigned __int8 *)(v5 + 64);
  *(_DWORD *)(a1 + 32) = a2;
  *(_BYTE *)(a1 + 44) = result;
  *(_DWORD *)(a1 + 24) = 1;
  return result;
}
