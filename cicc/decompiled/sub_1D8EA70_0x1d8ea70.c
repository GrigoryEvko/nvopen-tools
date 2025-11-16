// Function: sub_1D8EA70
// Address: 0x1d8ea70
//
__int64 __fastcall sub_1D8EA70(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rax
  _QWORD *v7; // rdx
  __int64 v8; // rcx
  unsigned __int64 v9; // r15
  _QWORD *v10; // rax
  _QWORD *v11; // rcx
  __int64 v12; // r14
  _QWORD *v13; // rdi
  _QWORD *v14; // r12
  __int64 (__fastcall *v15)(_QWORD *); // rdx
  __int64 v17; // [rsp+8h] [rbp-38h]

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
  v5 = a2;
  v6 = (v4 | (v4 >> 16) | HIDWORD(v4)) + 1;
  if ( v6 >= a2 )
    v5 = v6;
  if ( v5 > 0xFFFFFFFF )
    v5 = 0xFFFFFFFFLL;
  v17 = malloc(8 * v5);
  if ( !v17 )
    sub_16BD1C0("Allocation failed", 1u);
  v7 = *(_QWORD **)a1;
  v8 = 8LL * *(unsigned int *)(a1 + 8);
  v9 = *(_QWORD *)a1 + v8;
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = (_QWORD *)v17;
    v11 = (_QWORD *)(v17 + v8);
    do
    {
      if ( v10 )
      {
        *v10 = *v7;
        *v7 = 0;
      }
      ++v10;
      ++v7;
    }
    while ( v10 != v11 );
    v9 = *(_QWORD *)a1;
    v12 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( v12 != *(_QWORD *)a1 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v14 = *(_QWORD **)(v12 - 8);
          v12 -= 8;
          if ( v14 )
            break;
LABEL_19:
          if ( v9 == v12 )
            goto LABEL_23;
        }
        v15 = *(__int64 (__fastcall **)(_QWORD *))(*v14 + 8LL);
        if ( v15 == sub_1D59FF0 )
        {
          v13 = (_QWORD *)v14[1];
          *v14 = &unk_49F9CF0;
          if ( v13 != v14 + 3 )
            j_j___libc_free_0(v13, v14[3] + 1LL);
          j_j___libc_free_0(v14, 56);
          goto LABEL_19;
        }
        v15(v14);
        if ( v9 == v12 )
        {
LABEL_23:
          v9 = *(_QWORD *)a1;
          break;
        }
      }
    }
  }
  if ( v9 != a1 + 16 )
    _libc_free(v9);
  *(_DWORD *)(a1 + 12) = v5;
  *(_QWORD *)a1 = v17;
  return v17;
}
