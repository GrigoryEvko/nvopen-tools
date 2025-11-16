// Function: sub_398EE70
// Address: 0x398ee70
//
__int64 __fastcall sub_398EE70(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rax
  _QWORD *v6; // rdx
  __int64 v7; // rcx
  unsigned __int64 v8; // r12
  _QWORD *v9; // rax
  _QWORD *v10; // rcx
  __int64 v11; // rsi
  _QWORD *v12; // r14
  _QWORD *v13; // r15
  __int64 v15; // [rsp+8h] [rbp-38h]

  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v2 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v3 = ((v2
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | v2
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v4 = a2;
  v5 = (v3 | (v3 >> 16) | HIDWORD(v3)) + 1;
  if ( v5 >= a2 )
    v4 = v5;
  if ( v4 > 0xFFFFFFFF )
    v4 = 0xFFFFFFFFLL;
  v15 = malloc(16 * v4);
  if ( !v15 )
    sub_16BD1C0("Allocation failed", 1u);
  v6 = *(_QWORD **)a1;
  v7 = 16LL * *(unsigned int *)(a1 + 8);
  v8 = *(_QWORD *)a1 + v7;
  if ( *(_QWORD *)a1 != v8 )
  {
    v9 = (_QWORD *)v15;
    v10 = (_QWORD *)(v15 + v7);
    do
    {
      if ( v9 )
      {
        *v9 = *v6;
        v11 = v6[1];
        *v6 = 0;
        v9[1] = v11;
      }
      v9 += 2;
      v6 += 2;
    }
    while ( v9 != v10 );
    v12 = *(_QWORD **)a1;
    v8 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v8 )
    {
      do
      {
        v13 = *(_QWORD **)(v8 - 16);
        v8 -= 16LL;
        if ( v13 )
        {
          *v13 = &unk_4A3FCC0;
          sub_39A20E0(v13);
          j_j___libc_free_0((unsigned __int64)v13);
        }
      }
      while ( (_QWORD *)v8 != v12 );
      v8 = *(_QWORD *)a1;
    }
  }
  if ( v8 != a1 + 16 )
    _libc_free(v8);
  *(_DWORD *)(a1 + 12) = v4;
  *(_QWORD *)a1 = v15;
  return v15;
}
