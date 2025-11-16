// Function: sub_3862C60
// Address: 0x3862c60
//
__int64 __fastcall sub_3862C60(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  _QWORD *v7; // r12
  _QWORD *v8; // r15
  __int64 v9; // rbx
  __int64 v10; // rax
  _QWORD *v11; // rbx
  __int64 v12; // rax
  __int64 v14; // [rsp+8h] [rbp-38h]

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
  v14 = malloc(v6 << 6);
  if ( !v14 )
    sub_16BD1C0("Allocation failed", 1u);
  v7 = *(_QWORD **)a1;
  v8 = (_QWORD *)(*(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6));
  if ( *(_QWORD **)a1 != v8 )
  {
    v9 = v14;
    do
    {
      if ( v9 )
      {
        *(_QWORD *)v9 = 6;
        *(_QWORD *)(v9 + 8) = 0;
        v10 = v7[2];
        *(_QWORD *)(v9 + 16) = v10;
        if ( v10 != 0 && v10 != -8 && v10 != -16 )
          sub_1649AC0((unsigned __int64 *)v9, *v7 & 0xFFFFFFFFFFFFFFF8LL);
        *(_QWORD *)(v9 + 24) = v7[3];
        *(_QWORD *)(v9 + 32) = v7[4];
        *(_BYTE *)(v9 + 40) = *((_BYTE *)v7 + 40);
        *(_DWORD *)(v9 + 44) = *((_DWORD *)v7 + 11);
        *(_DWORD *)(v9 + 48) = *((_DWORD *)v7 + 12);
        *(_QWORD *)(v9 + 56) = v7[7];
      }
      v7 += 8;
      v9 += 64;
    }
    while ( v8 != v7 );
    v11 = *(_QWORD **)a1;
    v8 = (_QWORD *)(*(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6));
    if ( *(_QWORD **)a1 != v8 )
    {
      do
      {
        v12 = *(v8 - 6);
        v8 -= 8;
        if ( v12 != 0 && v12 != -8 && v12 != -16 )
          sub_1649B30(v8);
      }
      while ( v8 != v11 );
      v8 = *(_QWORD **)a1;
    }
  }
  if ( v8 != (_QWORD *)(a1 + 16) )
    _libc_free((unsigned __int64)v8);
  *(_DWORD *)(a1 + 12) = v6;
  *(_QWORD *)a1 = v14;
  return v14;
}
