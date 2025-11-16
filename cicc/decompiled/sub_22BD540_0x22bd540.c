// Function: sub_22BD540
// Address: 0x22bd540
//
__int64 __fastcall sub_22BD540(__int64 a1, __int64 a2)
{
  unsigned __int64 *v3; // r12
  unsigned __int64 *v4; // r15
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // r14
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int16 v9; // ax
  __int16 v10; // ax
  __int64 v11; // r13
  unsigned __int64 *v13; // [rsp+0h] [rbp-80h]
  char *v15[4]; // [rsp+10h] [rbp-70h] BYREF
  char *v16[10]; // [rsp+30h] [rbp-50h] BYREF

  if ( *(_BYTE *)(a1 + 336) )
  {
    v3 = *(unsigned __int64 **)(a1 + 312);
    v4 = *(unsigned __int64 **)(a1 + 320);
    v13 = v3;
    if ( v3 != v4 )
    {
      do
      {
        v5 = v3[1];
        v6 = *v3;
        if ( v5 != *v3 )
        {
          do
          {
            v7 = *(unsigned int *)(v6 + 144);
            v8 = *(_QWORD *)(v6 + 128);
            v6 += 152LL;
            sub_C7D6A0(v8, 8 * v7, 4);
            sub_C7D6A0(*(_QWORD *)(v6 - 56), 8LL * *(unsigned int *)(v6 - 40), 4);
            sub_C7D6A0(*(_QWORD *)(v6 - 88), 16LL * *(unsigned int *)(v6 - 72), 8);
            sub_C7D6A0(*(_QWORD *)(v6 - 120), 16LL * *(unsigned int *)(v6 - 104), 8);
          }
          while ( v5 != v6 );
          v6 = *v3;
        }
        if ( v6 )
          j_j___libc_free_0(v6);
        v3 += 3;
      }
      while ( v4 != v3 );
      *(_QWORD *)(a1 + 320) = v13;
    }
  }
  else
  {
    *(_QWORD *)(a1 + 312) = 0;
    *(_QWORD *)(a1 + 320) = 0;
    *(_QWORD *)(a1 + 328) = 0;
    *(_BYTE *)(a1 + 336) = 1;
  }
  v9 = *(_WORD *)(a1 + 304);
  v15[0] = 0;
  v15[1] = 0;
  *(_WORD *)(a1 + 296) = v9;
  LOBYTE(v9) = *(_BYTE *)(a1 + 306);
  v15[2] = 0;
  *(_BYTE *)(a1 + 267) = v9;
  v10 = *(_WORD *)(a1 + 307);
  v16[0] = 0;
  *(_WORD *)(a1 + 298) = v10;
  v16[1] = 0;
  v16[2] = 0;
  sub_22B6280(a1, a2, v15, v16);
  sub_22BADA0(a1, v15, (__int64 *)v16);
  v11 = a1 + 312;
  if ( v16[0] )
    j_j___libc_free_0((unsigned __int64)v16[0]);
  if ( v15[0] )
    j_j___libc_free_0((unsigned __int64)v15[0]);
  return v11;
}
