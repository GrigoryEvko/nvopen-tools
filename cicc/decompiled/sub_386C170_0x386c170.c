// Function: sub_386C170
// Address: 0x386c170
//
void __fastcall sub_386C170(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // r15
  _QWORD *v8; // rax
  _QWORD *v9; // r12
  unsigned __int64 *v10; // rbx
  unsigned __int64 v11; // rdx
  _QWORD *v12; // rbx
  __int64 v13; // rax
  _QWORD *v14; // [rsp+8h] [rbp-38h]

  v2 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v3 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v4 = ((v3
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | v3
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v5 = (v4 | (v4 >> 16) | HIDWORD(v4)) + 1;
  if ( v5 >= a2 )
    v2 = v5;
  v6 = v2;
  if ( v2 > 0xFFFFFFFF )
    v6 = 0xFFFFFFFFLL;
  v7 = malloc(24 * v6);
  if ( !v7 )
    sub_16BD1C0("Allocation failed", 1u);
  v8 = *(_QWORD **)a1;
  v9 = (_QWORD *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
  if ( *(_QWORD **)a1 != v9 )
  {
    v10 = (unsigned __int64 *)v7;
    do
    {
      if ( v10 )
      {
        *v10 = 6;
        v10[1] = 0;
        v11 = v8[2];
        v10[2] = v11;
        if ( v11 != 0 && v11 != -8 && v11 != -16 )
        {
          v14 = v8;
          sub_1649AC0(v10, *v8 & 0xFFFFFFFFFFFFFFF8LL);
          v8 = v14;
        }
      }
      v8 += 3;
      v10 += 3;
    }
    while ( v9 != v8 );
    v12 = *(_QWORD **)a1;
    v9 = (_QWORD *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
    if ( *(_QWORD **)a1 != v9 )
    {
      do
      {
        v13 = *(v9 - 1);
        v9 -= 3;
        if ( v13 != 0 && v13 != -8 && v13 != -16 )
          sub_1649B30(v9);
      }
      while ( v9 != v12 );
      v9 = *(_QWORD **)a1;
    }
  }
  if ( v9 != (_QWORD *)(a1 + 16) )
    _libc_free((unsigned __int64)v9);
  *(_QWORD *)a1 = v7;
  *(_DWORD *)(a1 + 12) = v6;
}
