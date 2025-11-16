// Function: sub_1B98CA0
// Address: 0x1b98ca0
//
__int64 __fastcall sub_1B98CA0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rax
  __int64 v4; // r13
  _QWORD *v5; // rdx
  __int64 v6; // rcx
  unsigned __int64 v7; // r14
  _QWORD *v8; // rcx
  _QWORD *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r15
  unsigned int v13; // [rsp+Ch] [rbp-34h]

  if ( a2 > 0xFFFFFFFF )
  {
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
LABEL_20:
    v4 = malloc(0x7FFFFFFF8uLL);
    if ( v4 )
    {
      v13 = -1;
      goto LABEL_6;
    }
    LODWORD(v2) = -1;
    goto LABEL_23;
  }
  v2 = a2;
  v3 = sub_1454B60(*(unsigned int *)(a1 + 12) + 2LL);
  if ( v3 >= a2 )
    v2 = v3;
  if ( v2 > 0xFFFFFFFF )
    goto LABEL_20;
  v13 = v2;
  v4 = malloc(8 * v2);
  if ( !v4 && (8 * v2 || (v4 = malloc(1u)) == 0) )
  {
LABEL_23:
    v4 = 0;
    sub_16BD1C0("Allocation failed", 1u);
    v13 = v2;
  }
LABEL_6:
  v5 = *(_QWORD **)a1;
  v6 = 8LL * *(unsigned int *)(a1 + 8);
  v7 = *(_QWORD *)a1 + v6;
  if ( *(_QWORD *)a1 != v7 )
  {
    v8 = (_QWORD *)(v4 + v6);
    v9 = (_QWORD *)v4;
    do
    {
      if ( v9 )
      {
        *v9 = *v5;
        *v5 = 0;
      }
      ++v9;
      ++v5;
    }
    while ( v9 != v8 );
    v7 = *(_QWORD *)a1;
    v10 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( v10 != *(_QWORD *)a1 )
    {
      do
      {
        v11 = *(_QWORD *)(v10 - 8);
        v10 -= 8;
        if ( v11 )
        {
          sub_1B949D0(v11);
          j_j___libc_free_0(v11, 472);
        }
      }
      while ( v7 != v10 );
      v7 = *(_QWORD *)a1;
    }
  }
  if ( v7 != a1 + 16 )
    _libc_free(v7);
  *(_QWORD *)a1 = v4;
  *(_DWORD *)(a1 + 12) = v13;
  return v13;
}
