// Function: sub_21762A0
// Address: 0x21762a0
//
__int64 __fastcall sub_21762A0(__int64 a1, unsigned int a2, _QWORD *a3)
{
  __int64 v5; // rsi
  _BYTE *v6; // rbx
  __int64 v7; // r13
  __int128 v9; // [rsp-10h] [rbp-40h]
  __int64 v10; // [rsp+0h] [rbp-30h] BYREF
  int v11; // [rsp+8h] [rbp-28h]

  v5 = *(_QWORD *)(a1 + 72);
  v10 = v5;
  if ( v5 )
    sub_1623A60((__int64)&v10, v5, 2);
  v6 = (_BYTE *)(*(_QWORD *)(a1 + 40) + 16LL * a2);
  v11 = *(_DWORD *)(a1 + 64);
  *((_QWORD *)&v9 + 1) = 1;
  *(_QWORD *)&v9 = *(_QWORD *)(a1 + 32);
  v7 = sub_1D25BD0(
         a3,
         (*v6 != 4) + 577,
         (__int64)&v10,
         (unsigned __int8)*v6,
         0,
         (unsigned int)(*v6 != 4) + 577,
         (unsigned __int8)*v6,
         v9);
  *(_DWORD *)(v7 + 64) = *(_DWORD *)(a1 + 64);
  if ( v10 )
    sub_161E7C0((__int64)&v10, v10);
  return v7;
}
