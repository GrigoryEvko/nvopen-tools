// Function: sub_3919EA0
// Address: 0x3919ea0
//
__int64 __fastcall sub_3919EA0(__int64 a1, _QWORD *a2)
{
  _QWORD *v4; // r13
  unsigned __int64 v5; // rax
  char v6; // dl
  unsigned __int64 v7; // rax
  __int64 v8; // r8
  char v9; // dl
  _BYTE *v10; // rsi
  unsigned int v11; // ecx
  _BYTE *v12; // rdi
  int v13; // esi
  _BYTE v15[48]; // [rsp+0h] [rbp-30h] BYREF

  v4 = *(_QWORD **)(a1 + 8);
  v5 = (*(__int64 (__fastcall **)(_QWORD *))(*v4 + 64LL))(v4) + v4[3] - v4[1] - a2[1];
  if ( v5 != (unsigned int)v5 )
    sub_16BD130("section size does not fit in a uint32_t", 1u);
  v6 = v5;
  v7 = v5 >> 7;
  v8 = 1;
  v9 = v6 & 0x7F;
  v10 = v15;
  v11 = 1;
  while ( 1 )
  {
    v12 = v10 + 1;
    *v10 = v9 | 0x80;
    if ( !v7 )
      break;
    ++v11;
    v9 = v7 & 0x7F;
    LOBYTE(v8) = v11 <= 4;
    v7 >>= 7;
    if ( !v7 && v11 > 4 )
    {
      *v12 = v9;
      v13 = (_DWORD)v10 + 2;
      return (*(__int64 (__fastcall **)(_QWORD, _BYTE *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 8) + 88LL))(
               *(_QWORD *)(a1 + 8),
               v15,
               v13 - (unsigned int)v15,
               *a2,
               v8);
    }
    ++v10;
  }
  if ( (_BYTE)v8 )
  {
    if ( v11 != 4 )
      v12 = (char *)memset(v12, 128, 4 - v11) + 4 - v11;
    *v12 = 0;
    v13 = (_DWORD)v12 + 1;
  }
  else
  {
    v13 = (_DWORD)v10 + 1;
  }
  return (*(__int64 (__fastcall **)(_QWORD, _BYTE *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 8) + 88LL))(
           *(_QWORD *)(a1 + 8),
           v15,
           v13 - (unsigned int)v15,
           *a2,
           v8);
}
