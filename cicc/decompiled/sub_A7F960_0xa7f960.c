// Function: sub_A7F960
// Address: 0xa7f960
//
__int64 __fastcall sub_A7F960(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, char a5)
{
  unsigned int v7; // ebx
  unsigned __int64 v8; // rcx
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  _BYTE *v11; // rax
  __int64 result; // rax
  __int64 v13; // rax
  int v14; // r9d
  __int64 v15; // r13
  unsigned int *v16; // rbx
  __int64 v17; // r12
  __int64 v18; // rdx
  __int64 v19; // rsi
  _QWORD v20[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v21; // [rsp+30h] [rbp-40h]

  v7 = 0;
  if ( a5 )
  {
    v7 = -1;
    v8 = sub_BCAE30(*(_QWORD *)(a3 + 8));
    v20[1] = v9;
    v20[0] = v8;
    if ( v8 > 7 )
    {
      _BitScanReverse64(&v10, v8 >> 3);
      v7 = 63 - (v10 ^ 0x3F);
    }
  }
  if ( *a4 <= 0x15u && (unsigned __int8)sub_AD7930(a4) )
  {
    v21 = 257;
    v13 = sub_BD2C40(80, unk_3F10A10);
    v15 = v13;
    if ( v13 )
      sub_B4D3C0(v13, a3, a2, 0, (unsigned __int8)v7, v14, 0, 0);
    result = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
               *(_QWORD *)(a1 + 88),
               v15,
               v20,
               *(_QWORD *)(a1 + 56),
               *(_QWORD *)(a1 + 64));
    v16 = *(unsigned int **)a1;
    v17 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v17 )
    {
      do
      {
        v18 = *((_QWORD *)v16 + 1);
        v19 = *v16;
        v16 += 4;
        result = sub_B99FD0(v15, v19, v18);
      }
      while ( (unsigned int *)v17 != v16 );
    }
  }
  else
  {
    v11 = sub_A7EC40(a1, (__int64)a4, *(_DWORD *)(*(_QWORD *)(a3 + 8) + 32LL));
    return sub_B34CE0(a1, a3, a2, v7, v11);
  }
  return result;
}
