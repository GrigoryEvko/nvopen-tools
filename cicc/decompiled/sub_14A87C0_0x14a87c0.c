// Function: sub_14A87C0
// Address: 0x14a87c0
//
void __fastcall sub_14A87C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // rbx
  __int64 i; // r15
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // [rsp+8h] [rbp-48h]

  v4 = *(_QWORD *)(a3 + 8);
  v5 = *(unsigned int *)(a2 + 8);
  for ( i = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a3 + 40) + 56LL) + 40LL); v4; v4 = *(_QWORD *)(v4 + 8) )
  {
    while ( 1 )
    {
      v7 = sub_1648700(v4);
      if ( *(_BYTE *)(v7 + 16) == 78 )
      {
        v8 = *(_QWORD *)(v7 - 24);
        if ( !*(_BYTE *)(v8 + 16) && *(_DWORD *)(v8 + 36) == 4 )
          break;
      }
      v4 = *(_QWORD *)(v4 + 8);
      if ( !v4 )
        goto LABEL_10;
    }
    if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v5 )
    {
      v10 = v7;
      sub_16CD150(a2, a2 + 16, 0, 8);
      v5 = *(unsigned int *)(a2 + 8);
      v7 = v10;
    }
    *(_QWORD *)(*(_QWORD *)a2 + 8 * v5) = v7;
    v5 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v5;
  }
LABEL_10:
  if ( (_DWORD)v5 )
  {
    v9 = sub_1649C60(*(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
    sub_14A8580(i, a1, v9, 0);
  }
}
