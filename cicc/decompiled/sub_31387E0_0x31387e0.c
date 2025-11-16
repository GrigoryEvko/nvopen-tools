// Function: sub_31387E0
// Address: 0x31387e0
//
bool __fastcall sub_31387E0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rsi
  __int64 v5; // rsi
  __int64 v6; // r9
  __int64 v7; // r14
  unsigned __int64 v8; // rsi
  _QWORD *v9; // rax
  int v10; // ecx
  _QWORD *v11; // rdx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rsi
  _QWORD v15[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = a1 + 512;
  v4 = *(_QWORD *)a2;
  if ( v4 )
  {
    sub_A88F30(a1 + 512, v4, *(_QWORD *)(a2 + 8), *(_WORD *)(a2 + 16));
  }
  else
  {
    *(_QWORD *)(a1 + 560) = 0;
    *(_QWORD *)(a1 + 568) = 0;
    *(_WORD *)(a1 + 576) = 0;
  }
  v5 = *(_QWORD *)(a2 + 24);
  v15[0] = v5;
  if ( !v5 || (sub_B96E90((__int64)v15, v5, 1), (v7 = v15[0]) == 0) )
  {
    sub_93FB40(v3, 0);
    v7 = v15[0];
    goto LABEL_13;
  }
  v8 = *(unsigned int *)(a1 + 520);
  v9 = *(_QWORD **)(a1 + 512);
  v10 = *(_DWORD *)(a1 + 520);
  v11 = &v9[2 * v8];
  if ( v9 == v11 )
  {
LABEL_16:
    v13 = *(unsigned int *)(a1 + 524);
    if ( v8 >= v13 )
    {
      v14 = v8 + 1;
      if ( v13 < v14 )
      {
        sub_C8D5F0(v3, (const void *)(a1 + 528), v14, 0x10u, a1 + 528, v6);
        v11 = (_QWORD *)(*(_QWORD *)(a1 + 512) + 16LL * *(unsigned int *)(a1 + 520));
      }
      *v11 = 0;
      v11[1] = v7;
      v7 = v15[0];
      ++*(_DWORD *)(a1 + 520);
    }
    else
    {
      if ( v11 )
      {
        *(_DWORD *)v11 = 0;
        v11[1] = v7;
        v7 = v15[0];
        v10 = *(_DWORD *)(a1 + 520);
      }
      *(_DWORD *)(a1 + 520) = v10 + 1;
    }
LABEL_13:
    if ( !v7 )
      return *(_QWORD *)a2 != 0;
    goto LABEL_10;
  }
  while ( *(_DWORD *)v9 )
  {
    v9 += 2;
    if ( v11 == v9 )
      goto LABEL_16;
  }
  v9[1] = v15[0];
LABEL_10:
  sub_B91220((__int64)v15, v7);
  return *(_QWORD *)a2 != 0;
}
