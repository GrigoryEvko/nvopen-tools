// Function: sub_388AB60
// Address: 0x388ab60
//
__int64 __fastcall sub_388AB60(__int64 a1, __int64 a2)
{
  __int64 *v4; // rdi
  unsigned int v5; // eax
  __int64 *v6; // rsi
  _QWORD *v7; // rdi
  __int64 result; // rax

  v4 = (__int64 *)(a1 + 32);
  *((_DWORD *)v4 - 8) = *(_DWORD *)a2;
  *(v4 - 3) = *(_QWORD *)(a2 + 8);
  *((_DWORD *)v4 - 4) = *(_DWORD *)(a2 + 16);
  *(v4 - 1) = *(_QWORD *)(a2 + 24);
  *(_QWORD *)(a1 + 32) = a1 + 48;
  sub_3887850(v4, *(_BYTE **)(a2 + 32), *(_QWORD *)(a2 + 32) + *(_QWORD *)(a2 + 40));
  *(_QWORD *)(a1 + 64) = a1 + 80;
  sub_3887850((__int64 *)(a1 + 64), *(_BYTE **)(a2 + 64), *(_QWORD *)(a2 + 64) + *(_QWORD *)(a2 + 72));
  v5 = *(_DWORD *)(a2 + 104);
  *(_DWORD *)(a1 + 104) = v5;
  if ( v5 > 0x40 )
    sub_16A4FD0(a1 + 96, (const void **)(a2 + 96));
  else
    *(_QWORD *)(a1 + 96) = *(_QWORD *)(a2 + 96);
  *(_BYTE *)(a1 + 108) = *(_BYTE *)(a2 + 108);
  v6 = (__int64 *)(a2 + 120);
  v7 = (_QWORD *)(a1 + 120);
  if ( *(void **)(a2 + 120) == sub_16982C0() )
    sub_169C6E0(v7, (__int64)v6);
  else
    sub_16986C0(v7, v6);
  result = *(_QWORD *)(a2 + 144);
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 144) = result;
  return result;
}
