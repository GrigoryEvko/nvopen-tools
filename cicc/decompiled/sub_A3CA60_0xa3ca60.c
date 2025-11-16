// Function: sub_A3CA60
// Address: 0xa3ca60
//
__int64 __fastcall sub_A3CA60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v6; // r13
  __int64 v7; // r12
  __int64 v8; // rdi
  __int64 v9; // rcx
  char v10; // al
  __int64 j; // r12
  __int64 v13; // rdi
  __int64 k; // r12
  __int64 v15; // rdi
  __int64 i; // [rsp+0h] [rbp-40h]

  v6 = *(_BYTE *)(a3 + 872);
  if ( v6 )
  {
    if ( unk_4F80E08 )
    {
      sub_BA8950(a3);
      v9 = 0;
      if ( !*(_BYTE *)(a2 + 9) )
        goto LABEL_9;
      goto LABEL_19;
    }
    v7 = *(_QWORD *)(a3 + 32);
    for ( i = a3 + 24; i != v7; v7 = *(_QWORD *)(v7 + 8) )
    {
      v8 = v7 - 56;
      if ( !v7 )
        v8 = 0;
      sub_B2B9A0(v8);
    }
    *(_BYTE *)(a3 + 872) = 0;
  }
  v9 = 0;
  if ( !*(_BYTE *)(a2 + 9) )
    goto LABEL_9;
LABEL_19:
  v9 = sub_BC0510(a4, &unk_4F87818, a3, 0) + 8;
LABEL_9:
  sub_A3ACE0(a3, *(_QWORD *)a2, *(_BYTE *)(a2 + 8), v9, *(_BYTE *)(a2 + 10), 0);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = &unk_4F82400;
  *(_QWORD *)a1 = 1;
  v10 = *(_BYTE *)(a3 + 872);
  if ( v6 )
  {
    if ( !v10 )
    {
      for ( j = *(_QWORD *)(a3 + 32); a3 + 24 != j; j = *(_QWORD *)(j + 8) )
      {
        v13 = j - 56;
        if ( !j )
          v13 = 0;
        sub_B2B950(v13);
      }
      *(_BYTE *)(a3 + 872) = 1;
    }
  }
  else if ( v10 )
  {
    for ( k = *(_QWORD *)(a3 + 32); a3 + 24 != k; k = *(_QWORD *)(k + 8) )
    {
      v15 = k - 56;
      if ( !k )
        v15 = 0;
      sub_B2B9A0(v15);
    }
    *(_BYTE *)(a3 + 872) = 0;
  }
  return a1;
}
