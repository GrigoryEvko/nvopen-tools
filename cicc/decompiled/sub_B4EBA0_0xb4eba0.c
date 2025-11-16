// Function: sub_B4EBA0
// Address: 0xb4eba0
//
__int64 __fastcall sub_B4EBA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v11; // rax
  __int64 v12; // rax
  bool v13; // zf
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 result; // rax
  __int64 v19; // [rsp+8h] [rbp-88h]
  _BYTE *v20; // [rsp+10h] [rbp-80h] BYREF
  __int64 v21; // [rsp+18h] [rbp-78h]
  _BYTE v22[112]; // [rsp+20h] [rbp-70h] BYREF

  v11 = *(_QWORD *)(a4 + 8);
  BYTE4(v19) = *(_BYTE *)(v11 + 8) == 18;
  LODWORD(v19) = *(_DWORD *)(v11 + 32);
  v12 = sub_BCE1B0(*(_QWORD *)(*(_QWORD *)(a2 + 8) + 24LL), v19);
  sub_B44260(a1, v12, 63, 2u, a7, a8);
  v13 = *(_QWORD *)(a1 - 64) == 0;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 80) = 0x400000000LL;
  if ( !v13 )
  {
    v14 = *(_QWORD *)(a1 - 56);
    **(_QWORD **)(a1 - 48) = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = *(_QWORD *)(a1 - 48);
  }
  v15 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a1 - 64) = a2;
  *(_QWORD *)(a1 - 56) = v15;
  if ( v15 )
    *(_QWORD *)(v15 + 16) = a1 - 56;
  v13 = *(_QWORD *)(a1 - 32) == 0;
  *(_QWORD *)(a1 - 48) = a2 + 16;
  *(_QWORD *)(a2 + 16) = a1 - 64;
  if ( !v13 )
  {
    v16 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a3;
  if ( a3 )
  {
    v17 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(a1 - 24) = v17;
    if ( v17 )
      *(_QWORD *)(v17 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a3 + 16;
    *(_QWORD *)(a3 + 16) = a1 - 32;
  }
  v21 = 0x1000000000LL;
  v20 = v22;
  sub_B4E3E0((unsigned __int8 *)a4, (__int64 *)&v20);
  sub_B4E7F0(a1, v20, (unsigned int)v21);
  result = sub_BD6B50(a1, a5);
  if ( v20 != v22 )
    return _libc_free(v20, a5);
  return result;
}
