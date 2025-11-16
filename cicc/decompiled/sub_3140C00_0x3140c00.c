// Function: sub_3140C00
// Address: 0x3140c00
//
__int64 __fastcall sub_3140C00(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // r12
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 v11; // rsi
  __int64 v12; // rdi

  v1 = (unsigned int)(*(_DWORD *)(a1 + 24) - 1);
  *(_DWORD *)(a1 + 24) = v1;
  v2 = *(_QWORD *)(a1 + 16) + 32 * v1;
  v3 = *(unsigned int *)(v2 + 24);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)(v2 + 8);
    v5 = v4 + 72 * v3;
    do
    {
      if ( *(_QWORD *)v4 != -8192 && *(_QWORD *)v4 != -4096 )
      {
        sub_C7D6A0(*(_QWORD *)(v4 + 48), 24LL * *(unsigned int *)(v4 + 64), 8);
        sub_C7D6A0(*(_QWORD *)(v4 + 16), 24LL * *(unsigned int *)(v4 + 32), 8);
      }
      v4 += 72;
    }
    while ( v5 != v4 );
    v3 = *(unsigned int *)(v2 + 24);
  }
  sub_C7D6A0(*(_QWORD *)(v2 + 8), 72 * v3, 8);
  v6 = (unsigned int)(*(_DWORD *)(a1 + 104) - 1);
  *(_DWORD *)(a1 + 104) = v6;
  v7 = *(_QWORD *)(a1 + 96) + 32 * v6;
  v8 = *(unsigned int *)(v7 + 24);
  if ( (_DWORD)v8 )
  {
    v9 = *(_QWORD *)(v7 + 8);
    v10 = v9 + 48 * v8;
    do
    {
      while ( *(_QWORD *)v9 == -1 || *(_QWORD *)v9 == -2 )
      {
        v9 += 48;
        if ( v10 == v9 )
          goto LABEL_14;
      }
      v11 = *(unsigned int *)(v9 + 40);
      v12 = *(_QWORD *)(v9 + 24);
      v9 += 48;
      sub_C7D6A0(v12, 32 * v11, 8);
    }
    while ( v10 != v9 );
LABEL_14:
    v8 = *(unsigned int *)(v7 + 24);
  }
  return sub_C7D6A0(*(_QWORD *)(v7 + 8), 48 * v8, 8);
}
