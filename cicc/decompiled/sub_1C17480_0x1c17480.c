// Function: sub_1C17480
// Address: 0x1c17480
//
__int64 __fastcall sub_1C17480(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  _BYTE *v7; // r13
  __int64 v9; // rsi
  __int64 v10; // rax
  _BYTE *v11; // rdi
  __int64 v12; // rdx
  size_t v13; // rcx
  __int64 v14; // rsi
  _QWORD *v15; // rdi
  __int64 result; // rax
  size_t v17; // rdx
  __int64 v18; // [rsp+0h] [rbp-80h] BYREF
  __int16 v19; // [rsp+10h] [rbp-70h]
  _QWORD *v20; // [rsp+20h] [rbp-60h] BYREF
  size_t n; // [rsp+28h] [rbp-58h]
  _QWORD src[4]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v23; // [rsp+50h] [rbp-30h]

  v7 = (_BYTE *)(a1 + 96);
  *(_QWORD *)(a1 + 16) = a2;
  v9 = a2 + 240;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 24) = a4;
  *(_QWORD *)(a1 + 32) = a3;
  *(_DWORD *)(a1 + 72) = a5;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0;
  *(_BYTE *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)a1 = &unk_49F7608;
  *(_QWORD *)(a1 + 144) = 0;
  *(_BYTE *)(a1 + 136) = a7;
  v10 = *(_QWORD *)(v9 - 240);
  *(_BYTE *)(a1 + 137) = a6;
  *(_QWORD *)(a1 + 8) = v10;
  *(_QWORD *)(a1 + 40) = 0x4100000001LL;
  *(_QWORD *)(a1 + 48) = 0x6200000002LL;
  *(_QWORD *)(a1 + 56) = 0x200000003LL;
  v19 = 260;
  *(_DWORD *)(a1 + 64) = 7;
  v18 = v9;
  sub_16E1010((__int64)&v20, (__int64)&v18);
  v11 = *(_BYTE **)(a1 + 80);
  if ( v20 == src )
  {
    v17 = n;
    if ( n )
    {
      if ( n == 1 )
        *v11 = src[0];
      else
        memcpy(v11, src, n);
      v17 = n;
      v11 = *(_BYTE **)(a1 + 80);
    }
    *(_QWORD *)(a1 + 88) = v17;
    v11[v17] = 0;
    v11 = v20;
  }
  else
  {
    v12 = src[0];
    v13 = n;
    if ( v7 == v11 )
    {
      *(_QWORD *)(a1 + 80) = v20;
      *(_QWORD *)(a1 + 88) = v13;
      *(_QWORD *)(a1 + 96) = v12;
    }
    else
    {
      v14 = *(_QWORD *)(a1 + 96);
      *(_QWORD *)(a1 + 80) = v20;
      *(_QWORD *)(a1 + 88) = v13;
      *(_QWORD *)(a1 + 96) = v12;
      if ( v11 )
      {
        v20 = v11;
        src[0] = v14;
        goto LABEL_5;
      }
    }
    v20 = src;
    v11 = src;
  }
LABEL_5:
  n = 0;
  *v11 = 0;
  v15 = v20;
  *(_QWORD *)(a1 + 112) = src[2];
  *(_QWORD *)(a1 + 120) = src[3];
  result = v23;
  *(_QWORD *)(a1 + 128) = v23;
  if ( v15 != src )
    return j_j___libc_free_0(v15, src[0] + 1LL);
  return result;
}
