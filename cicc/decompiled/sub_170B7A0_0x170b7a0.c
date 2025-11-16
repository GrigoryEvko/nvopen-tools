// Function: sub_170B7A0
// Address: 0x170b7a0
//
__int64 __fastcall sub_170B7A0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  _QWORD *v7; // rax
  _QWORD *v8; // r12
  _QWORD *v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rdx
  _QWORD *v12; // rbx
  __int64 v13; // rax
  __int64 v15; // [rsp+0h] [rbp-40h]
  _QWORD *v16; // [rsp+8h] [rbp-38h]

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
  v15 = malloc(40 * v6);
  if ( !v15 )
    sub_16BD1C0("Allocation failed", 1u);
  v7 = *(_QWORD **)a1;
  v8 = (_QWORD *)(*(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8));
  if ( *(_QWORD **)a1 != v8 )
  {
    v9 = (_QWORD *)v15;
    do
    {
      if ( v9 )
      {
        v10 = v7[1];
        v9[2] = 0;
        v9[1] = v10 & 6;
        v11 = v7[3];
        v9[3] = v11;
        if ( v11 != 0 && v11 != -8 && v11 != -16 )
        {
          v16 = v7;
          sub_1649AC0(v9 + 1, v7[1] & 0xFFFFFFFFFFFFFFF8LL);
          v7 = v16;
        }
        *v9 = off_49EFFB0;
        v9[4] = v7[4];
      }
      v7 += 5;
      v9 += 5;
    }
    while ( v8 != v7 );
    v12 = *(_QWORD **)a1;
    v8 = (_QWORD *)(*(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8));
    if ( *(_QWORD **)a1 != v8 )
    {
      do
      {
        v13 = *(v8 - 2);
        v8 -= 5;
        *v8 = &unk_49EE2B0;
        if ( v13 != 0 && v13 != -8 && v13 != -16 )
          sub_1649B30(v8 + 1);
      }
      while ( v8 != v12 );
      v8 = *(_QWORD **)a1;
    }
  }
  if ( v8 != (_QWORD *)(a1 + 16) )
    _libc_free((unsigned __int64)v8);
  *(_DWORD *)(a1 + 12) = v6;
  *(_QWORD *)a1 = v15;
  return v15;
}
