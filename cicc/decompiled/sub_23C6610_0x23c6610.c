// Function: sub_23C6610
// Address: 0x23c6610
//
__int64 __fastcall sub_23C6610(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  volatile signed __int32 *v4; // rdi

  sub_2240AE0((unsigned __int64 *)a1, (unsigned __int64 *)a2);
  sub_2240AE0((unsigned __int64 *)(a1 + 32), (unsigned __int64 *)(a2 + 32));
  sub_2240AE0((unsigned __int64 *)(a1 + 64), (unsigned __int64 *)(a2 + 64));
  sub_2240AE0((unsigned __int64 *)(a1 + 96), (unsigned __int64 *)(a2 + 96));
  *(_DWORD *)(a1 + 128) = *(_DWORD *)(a2 + 128);
  *(_DWORD *)(a1 + 132) = *(_DWORD *)(a2 + 132);
  *(_DWORD *)(a1 + 136) = *(_DWORD *)(a2 + 136);
  *(_BYTE *)(a1 + 140) = *(_BYTE *)(a2 + 140);
  *(_BYTE *)(a1 + 141) = *(_BYTE *)(a2 + 141);
  *(_BYTE *)(a1 + 142) = *(_BYTE *)(a2 + 142);
  v3 = *(_QWORD *)(a2 + 144);
  if ( v3 )
    _InterlockedAdd((volatile signed __int32 *)(v3 + 8), 1u);
  v4 = *(volatile signed __int32 **)(a1 + 144);
  *(_QWORD *)(a1 + 144) = v3;
  if ( !v4 || _InterlockedSub(v4 + 2, 1u) )
    return a1;
  (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v4 + 8LL))(v4);
  return a1;
}
