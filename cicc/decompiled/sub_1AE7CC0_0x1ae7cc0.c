// Function: sub_1AE7CC0
// Address: 0x1ae7cc0
//
__int64 __fastcall sub_1AE7CC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdx
  _BYTE *v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rax
  _QWORD *v14; // rdi
  int v15; // [rsp+8h] [rbp-78h] BYREF
  char v16; // [rsp+Ch] [rbp-74h]
  _QWORD *v17; // [rsp+10h] [rbp-70h]
  __int64 v18; // [rsp+18h] [rbp-68h]
  _QWORD v19[12]; // [rsp+20h] [rbp-60h] BYREF

  v6 = *(_QWORD *)(*(_QWORD *)(a3 + 24 * (1LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF))) + 24LL);
  v7 = *(_BYTE **)(v6 + 8 * (3LL - *(unsigned int *)(v6 + 8)));
  if ( *v7 == 11 && (sub_15B0CD0((__int64)&v15, (__int64)v7), v16) )
  {
    v8 = *(_QWORD *)(a3 + 24 * (2LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
    if ( v15 )
    {
      v9 = *(_QWORD *)(v8 + 24);
      *(_BYTE *)(a1 + 8) = 1;
      *(_QWORD *)a1 = v9;
      return a1;
    }
    v11 = **(_QWORD **)(a2 + 8);
    v19[3] = 37;
    v19[4] = 48;
    v19[5] = 32;
    v19[2] = v11 - 1;
    v19[6] = 30;
    v19[7] = 33;
    v19[1] = 16;
    v12 = *(_QWORD *)(v8 + 24);
    v17 = v19;
    v18 = 0x800000008LL;
    v19[0] = 18;
    v13 = sub_15C4CE0(v12, v19, 8);
    v14 = v17;
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = v13;
    if ( v14 != v19 )
      _libc_free((unsigned __int64)v14);
  }
  else
  {
    *(_BYTE *)(a1 + 8) = 0;
  }
  return a1;
}
