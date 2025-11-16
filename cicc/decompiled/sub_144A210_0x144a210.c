// Function: sub_144A210
// Address: 0x144a210
//
__int64 __fastcall sub_144A210(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rdx
  __int64 *v6; // rsi
  __int64 v7; // rdi
  unsigned int v8; // esi
  __int64 *v9; // rcx
  __int64 v10; // r10
  int v12; // ecx
  int v13; // r11d
  __int64 v14; // [rsp+0h] [rbp-30h] BYREF
  __int64 v15; // [rsp+8h] [rbp-28h]
  __int64 v16; // [rsp+10h] [rbp-20h]
  int v17; // [rsp+18h] [rbp-18h]

  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  sub_1449800(a1, a2, (__int64)&v14);
  v3 = *(_QWORD *)(a2 + 80);
  v4 = *(_QWORD *)(a1 + 8);
  if ( v3 )
    v3 -= 24;
  v5 = *(unsigned int *)(v4 + 48);
  v6 = 0;
  if ( (_DWORD)v5 )
  {
    v7 = *(_QWORD *)(v4 + 32);
    v8 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v9 = (__int64 *)(v7 + 16LL * v8);
    v10 = *v9;
    if ( v3 == *v9 )
    {
LABEL_5:
      if ( v9 != (__int64 *)(v7 + 16 * v5) )
      {
        v6 = (__int64 *)v9[1];
        goto LABEL_7;
      }
    }
    else
    {
      v12 = 1;
      while ( v10 != -8 )
      {
        v13 = v12 + 1;
        v8 = (v5 - 1) & (v12 + v8);
        v9 = (__int64 *)(v7 + 16LL * v8);
        v10 = *v9;
        if ( v3 == *v9 )
          goto LABEL_5;
        v12 = v13;
      }
    }
    v6 = 0;
  }
LABEL_7:
  sub_1449060(a1, v6, *(_QWORD **)(a1 + 32));
  return j___libc_free_0(v15);
}
