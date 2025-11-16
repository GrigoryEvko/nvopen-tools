// Function: sub_1E690F0
// Address: 0x1e690f0
//
__int64 __fastcall sub_1E690F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  _QWORD *v4; // r13
  __int64 v5; // rdx
  __int64 *v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rsi
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // r8
  int v13; // edx
  int v14; // r9d
  __int64 v15; // [rsp+0h] [rbp-40h] BYREF
  __int64 v16; // [rsp+8h] [rbp-38h]
  __int64 v17; // [rsp+10h] [rbp-30h]
  int v18; // [rsp+18h] [rbp-28h]

  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  sub_1E686E0(a1, a2, (__int64)&v15);
  v2 = *(_QWORD *)(a2 + 328);
  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(_QWORD **)(a1 + 32);
  sub_1E06620(v3);
  v5 = *(_QWORD *)(v3 + 1312);
  v6 = 0;
  v7 = *(unsigned int *)(v5 + 48);
  if ( (_DWORD)v7 )
  {
    v8 = *(_QWORD *)(v5 + 32);
    v9 = (v7 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
    v10 = (__int64 *)(v8 + 16LL * v9);
    v11 = *v10;
    if ( v2 == *v10 )
    {
LABEL_3:
      if ( v10 != (__int64 *)(v8 + 16 * v7) )
      {
        v6 = (__int64 *)v10[1];
        goto LABEL_5;
      }
    }
    else
    {
      v13 = 1;
      while ( v11 != -8 )
      {
        v14 = v13 + 1;
        v9 = (v7 - 1) & (v13 + v9);
        v10 = (__int64 *)(v8 + 16LL * v9);
        v11 = *v10;
        if ( v2 == *v10 )
          goto LABEL_3;
        v13 = v14;
      }
    }
    v6 = 0;
  }
LABEL_5:
  sub_1E67F10(a1, v6, v4);
  return j___libc_free_0(v16);
}
