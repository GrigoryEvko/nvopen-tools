// Function: sub_D345D0
// Address: 0xd345d0
//
__int64 __fastcall sub_D345D0(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v3; // r10
  __int64 v4; // rax
  __int64 v5; // rdx

  v3 = *(_QWORD *)(a3 + 288);
  v4 = *(_QWORD *)(a3 + 8) + 72LL * a2;
  v5 = *(_QWORD *)(*(_QWORD *)(v4 + 16) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
    v5 = **(_QWORD **)(v5 + 16);
  return sub_D344F0(
           a1,
           a2,
           *(_QWORD *)(v4 + 24),
           *(_QWORD *)(v4 + 32),
           *(_DWORD *)(v5 + 8) >> 8,
           *(_BYTE *)(v4 + 64),
           v3);
}
