// Function: sub_2468350
// Address: 0x2468350
//
void __fastcall sub_2468350(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rsi
  __int64 v6; // rsi
  __int64 v7; // r9
  __int64 v8; // r13
  unsigned __int64 v9; // rsi
  _QWORD *v10; // rax
  int v11; // ecx
  _QWORD *v12; // rdx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rsi
  _QWORD v15[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = a2[5];
  v4 = a2[4];
  if ( v4 == v3 + 48 || !v4 )
    v5 = 0;
  else
    v5 = v4 - 24;
  sub_23D0AB0(a1, v5, 0, 0, 0);
  v6 = a2[6];
  v15[0] = v6;
  if ( v6 && (sub_B96E90((__int64)v15, v6, 1), (v8 = v15[0]) != 0) )
  {
    v9 = *(unsigned int *)(a1 + 8);
    v10 = *(_QWORD **)a1;
    v11 = *(_DWORD *)(a1 + 8);
    v12 = (_QWORD *)(*(_QWORD *)a1 + 16 * v9);
    if ( *(_QWORD **)a1 != v12 )
    {
      while ( *(_DWORD *)v10 )
      {
        v10 += 2;
        if ( v12 == v10 )
          goto LABEL_16;
      }
      v10[1] = v15[0];
      goto LABEL_11;
    }
LABEL_16:
    v13 = *(unsigned int *)(a1 + 12);
    if ( v9 >= v13 )
    {
      v14 = v9 + 1;
      if ( v13 < v14 )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v14, 0x10u, a1 + 16, v7);
        v12 = (_QWORD *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
      }
      *v12 = 0;
      v12[1] = v8;
      v8 = v15[0];
      ++*(_DWORD *)(a1 + 8);
    }
    else
    {
      if ( v12 )
      {
        *(_DWORD *)v12 = 0;
        v12[1] = v8;
        v8 = v15[0];
        v11 = *(_DWORD *)(a1 + 8);
      }
      *(_DWORD *)(a1 + 8) = v11 + 1;
    }
  }
  else
  {
    sub_93FB40(a1, 0);
    v8 = v15[0];
  }
  if ( v8 )
LABEL_11:
    sub_B91220((__int64)v15, v8);
}
