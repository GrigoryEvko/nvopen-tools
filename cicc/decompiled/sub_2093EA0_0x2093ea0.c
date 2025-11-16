// Function: sub_2093EA0
// Address: 0x2093ea0
//
void __fastcall sub_2093EA0(__int64 a1, __int64 *a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v7; // r13
  int v8; // edx
  char v9; // al
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rdi
  __int64 **v13; // r8
  int v14; // edx
  unsigned int v15; // eax
  __int64 **v16; // rsi
  __int64 *v17; // r9
  __int64 v18; // rdi
  bool v19; // al
  bool v20; // r11
  bool v21; // r10
  bool v22; // r9
  __int16 v23; // r8
  __int16 v24; // cx
  int v25; // eax
  __int64 v26; // rdx
  int v27; // eax
  __int64 v28; // rax
  int v29; // esi
  int v30; // r10d
  __int64 *v31; // [rsp+8h] [rbp-28h] BYREF

  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 176LL);
  v8 = *((unsigned __int8 *)a2 + 16);
  if ( (unsigned int)(v8 - 25) <= 9 )
  {
    sub_2092EF0(a1, a2[5], a3, a4, a5);
    if ( *((_BYTE *)a2 + 16) != 78 )
      goto LABEL_3;
  }
  else if ( (_BYTE)v8 != 78 )
  {
LABEL_3:
    ++*(_DWORD *)(a1 + 536);
    goto LABEL_4;
  }
  v28 = *(a2 - 3);
  if ( *(_BYTE *)(v28 + 16) || (*(_BYTE *)(v28 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v28 + 36) - 35) > 3 )
    goto LABEL_3;
LABEL_4:
  *(_QWORD *)a1 = a2;
  sub_2067240((__int64 *)a1, *((unsigned __int8 *)a2 + 16) - 24, (__int64)a2, a3, *(double *)a4.m128i_i64, a5);
  v9 = *(_BYTE *)(*a2 + 8);
  if ( v9 == 16 )
    v9 = *(_BYTE *)(**(_QWORD **)(*a2 + 16) + 8LL);
  if ( (unsigned __int8)(v9 - 1) <= 5u || (v25 = *((unsigned __int8 *)a2 + 16), (_BYTE)v25 == 76) )
  {
    v10 = *(unsigned int *)(a1 + 32);
    v11 = *(_QWORD *)(a1 + 16);
    v31 = a2;
    v12 = a1 + 8;
    v13 = (__int64 **)(v11 + 24 * v10);
    if ( (_DWORD)v10 )
    {
      v14 = v10 - 1;
      v15 = v14 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v16 = (__int64 **)(v11 + 24LL * v15);
      v17 = *v16;
      if ( a2 == *v16 )
      {
LABEL_9:
        if ( v13 != v16 )
        {
          v18 = sub_205F5C0(v12, (__int64 *)&v31)[1];
          if ( v18 )
          {
            v24 = *((unsigned __int8 *)a2 + 17);
            v19 = (v24 & 4) != 0;
            v20 = (*((_BYTE *)a2 + 17) & 8) != 0;
            v21 = (*((_BYTE *)a2 + 17) & 0x10) != 0;
            v22 = (*((_BYTE *)a2 + 17) & 0x20) != 0;
            v23 = (*((_BYTE *)a2 + 17) & 0x40) != 0;
            LOBYTE(v24) = (unsigned __int8)v24 >> 7;
            if ( (*(_BYTE *)(v18 + 80) & 1) != 0 )
              sub_1D19330(
                v18,
                (16 * v19)
              | 1
              | (32 * v20)
              | (v21 << 6)
              | (v22 << 7)
              | (v23 << 9)
              | (v24 << 10)
              | (((*((_BYTE *)a2 + 17) & 2) != 0) << 11));
            else
              *(_WORD *)(v18 + 80) = *(_WORD *)(v18 + 80) & 0xF000
                                   | (((*((_BYTE *)a2 + 17) & 2) != 0) << 11)
                                   | (v24 << 10)
                                   | (v23 << 9)
                                   | (v21 << 6)
                                   | (32 * v20)
                                   | (16 * v19)
                                   | 1
                                   | (v22 << 7);
          }
        }
      }
      else
      {
        v29 = 1;
        while ( v17 != (__int64 *)-8LL )
        {
          v30 = v29 + 1;
          v15 = v14 & (v29 + v15);
          v16 = (__int64 **)(v11 + 24LL * v15);
          v17 = *v16;
          if ( a2 == *v16 )
            goto LABEL_9;
          v29 = v30;
        }
      }
    }
    v25 = *((unsigned __int8 *)a2 + 16);
  }
  if ( (unsigned int)(v25 - 25) > 9 && !*(_BYTE *)(a1 + 760) && !sub_1642D30((__int64)a2) )
    sub_208C7F0(a1, a2, a3, a4, a5);
  v26 = *(_QWORD *)(a1 + 552);
  if ( v7 != *(_QWORD *)(v26 + 176) )
  {
    v27 = *(_DWORD *)(a1 + 536) + 1;
    *(_DWORD *)(a1 + 536) = v27;
    *(_DWORD *)(*(_QWORD *)(v26 + 176) + 64LL) = v27;
  }
  *(_QWORD *)a1 = 0;
}
