// Function: sub_28E92A0
// Address: 0x28e92a0
//
__int64 __fastcall sub_28E92A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5, char a6, __int64 a7)
{
  __int64 v10; // rdi
  int v11; // esi
  __int64 v12; // rdx
  __int64 v14; // r12
  int v15; // esi

  v10 = *(_QWORD *)(a1 + 8);
  v11 = *(unsigned __int8 *)(v10 + 8);
  if ( (unsigned int)(v11 - 17) <= 1 )
    LOBYTE(v11) = *(_BYTE *)(**(_QWORD **)(v10 + 16) + 8LL);
  v12 = a5;
  BYTE1(v12) = a6;
  if ( (_BYTE)v11 == 12 )
    return sub_B504D0(17, a1, a2, a3, a4, v12);
  v14 = sub_B504D0(18, a1, a2, a3, a4, v12);
  v15 = *(_BYTE *)(a7 + 1) >> 1;
  if ( v15 == 127 )
    v15 = -1;
  sub_B45150(v14, v15);
  return v14;
}
