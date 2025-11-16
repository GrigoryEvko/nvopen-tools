// Function: sub_10BF960
// Address: 0x10bf960
//
void __fastcall sub_10BF960(__int64 a1, __int64 a2, __int16 a3)
{
  __int64 v3; // r9
  __int64 v4; // r13
  unsigned __int64 v5; // rsi
  _QWORD *v6; // rax
  int v7; // ecx
  _QWORD *v8; // rdx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rsi
  __int64 v11[6]; // [rsp-30h] [rbp-30h] BYREF

  if ( !a2 )
    BUG();
  *(_QWORD *)(a1 + 48) = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a1 + 56) = a2;
  *(_WORD *)(a1 + 64) = a3;
  v11[0] = *(_QWORD *)sub_B46C60(a2 - 24);
  if ( v11[0] && (sub_B96E90((__int64)v11, v11[0], 1), (v4 = v11[0]) != 0) )
  {
    v5 = *(unsigned int *)(a1 + 8);
    v6 = *(_QWORD **)a1;
    v7 = *(_DWORD *)(a1 + 8);
    v8 = (_QWORD *)(*(_QWORD *)a1 + 16 * v5);
    if ( *(_QWORD **)a1 != v8 )
    {
      while ( *(_DWORD *)v6 )
      {
        v6 += 2;
        if ( v8 == v6 )
          goto LABEL_14;
      }
      v6[1] = v11[0];
      goto LABEL_9;
    }
LABEL_14:
    v9 = *(unsigned int *)(a1 + 12);
    if ( v5 >= v9 )
    {
      v10 = v5 + 1;
      if ( v9 < v10 )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v10, 0x10u, a1 + 16, v3);
        v8 = (_QWORD *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
      }
      *v8 = 0;
      v8[1] = v4;
      v4 = v11[0];
      ++*(_DWORD *)(a1 + 8);
    }
    else
    {
      if ( v8 )
      {
        *(_DWORD *)v8 = 0;
        v8[1] = v4;
        v4 = v11[0];
        v7 = *(_DWORD *)(a1 + 8);
      }
      *(_DWORD *)(a1 + 8) = v7 + 1;
    }
  }
  else
  {
    sub_93FB40(a1, 0);
    v4 = v11[0];
  }
  if ( v4 )
LABEL_9:
    sub_B91220((__int64)v11, v4);
}
