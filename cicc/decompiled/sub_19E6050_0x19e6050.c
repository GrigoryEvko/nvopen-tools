// Function: sub_19E6050
// Address: 0x19e6050
//
__int64 __fastcall sub_19E6050(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rax
  __int64 v5; // r14
  unsigned __int64 v6; // r12
  __int64 v7; // r15
  _QWORD *v8; // rbx
  __int64 v9; // rbx
  unsigned __int64 v10; // rdi
  unsigned int v12; // [rsp+Ch] [rbp-34h]

  if ( a2 > 0xFFFFFFFF )
  {
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
LABEL_20:
    v5 = malloc(0x67FFFFFF98uLL);
    if ( v5 )
    {
      v12 = -1;
      goto LABEL_6;
    }
    LODWORD(v3) = -1;
    goto LABEL_23;
  }
  v3 = a2;
  v4 = sub_1454B60(*(unsigned int *)(a1 + 12) + 2LL);
  if ( v4 >= a2 )
    v3 = v4;
  if ( v3 > 0xFFFFFFFF )
    goto LABEL_20;
  v12 = v3;
  v5 = malloc(104 * v3);
  if ( !v5 && (104 * v3 || (v5 = malloc(1u)) == 0) )
  {
LABEL_23:
    v5 = 0;
    sub_16BD1C0("Allocation failed", 1u);
    v12 = v3;
  }
LABEL_6:
  v6 = *(_QWORD *)a1 + 104LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v6 )
  {
    v7 = *(_QWORD *)a1;
    v8 = (_QWORD *)v5;
    do
    {
      if ( v8 )
        sub_16CCEE0(v8, (__int64)(v8 + 5), 8, v7);
      v7 += 104;
      v8 += 13;
    }
    while ( v6 != v7 );
    v9 = *(_QWORD *)a1;
    v6 = *(_QWORD *)a1 + 104LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v6 )
    {
      do
      {
        v6 -= 104LL;
        v10 = *(_QWORD *)(v6 + 16);
        if ( v10 != *(_QWORD *)(v6 + 8) )
          _libc_free(v10);
      }
      while ( v6 != v9 );
      v6 = *(_QWORD *)a1;
    }
  }
  if ( v6 != a1 + 16 )
    _libc_free(v6);
  *(_QWORD *)a1 = v5;
  *(_DWORD *)(a1 + 12) = v12;
  return v12;
}
