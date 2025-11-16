// Function: sub_15F6E70
// Address: 0x15f6e70
//
__int64 __fastcall sub_15F6E70(__int64 a1, __int64 a2)
{
  unsigned int v3; // r13d
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned int v6; // eax
  __int64 result; // rax
  _QWORD *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned __int64 v11; // rcx
  __int64 v12; // rcx

  v3 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v4 = sub_16498A0(a2);
  v5 = sub_1643270(v4);
  sub_15F1EA0(a1, v5, 1, a1 - 24LL * v3, v3, 0);
  v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( v6 )
  {
    v8 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    v9 = *(_QWORD *)(a2 - 24LL * v6);
    if ( *v8 )
    {
      v10 = v8[1];
      v11 = v8[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v11 = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = *(_QWORD *)(v10 + 16) & 3LL | v11;
    }
    *v8 = v9;
    if ( v9 )
    {
      v12 = *(_QWORD *)(v9 + 8);
      v8[1] = v12;
      if ( v12 )
        *(_QWORD *)(v12 + 16) = (unsigned __int64)(v8 + 1) | *(_QWORD *)(v12 + 16) & 3LL;
      v8[2] = (v9 + 8) | v8[2] & 3LL;
      *(_QWORD *)(v9 + 8) = v8;
    }
  }
  result = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1u;
  *(_BYTE *)(a1 + 17) = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1;
  return result;
}
