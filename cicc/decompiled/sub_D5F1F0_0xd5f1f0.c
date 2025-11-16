// Function: sub_D5F1F0
// Address: 0xd5f1f0
//
void __fastcall sub_D5F1F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rsi
  __int64 v4; // r9
  __int64 v5; // r13
  unsigned __int64 v6; // rsi
  _QWORD *v7; // rax
  int v8; // ecx
  _QWORD *v9; // rdx
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rsi
  _QWORD v12[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(_QWORD *)(a2 + 40);
  *(_WORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 48) = v2;
  *(_QWORD *)(a1 + 56) = a2 + 24;
  v3 = *(_QWORD *)sub_B46C60(a2);
  v12[0] = v3;
  if ( v3 && (sub_B96E90((__int64)v12, v3, 1), (v5 = v12[0]) != 0) )
  {
    v6 = *(unsigned int *)(a1 + 8);
    v7 = *(_QWORD **)a1;
    v8 = *(_DWORD *)(a1 + 8);
    v9 = (_QWORD *)(*(_QWORD *)a1 + 16 * v6);
    if ( *(_QWORD **)a1 != v9 )
    {
      while ( *(_DWORD *)v7 )
      {
        v7 += 2;
        if ( v9 == v7 )
          goto LABEL_13;
      }
      v7[1] = v12[0];
      goto LABEL_8;
    }
LABEL_13:
    v10 = *(unsigned int *)(a1 + 12);
    if ( v6 >= v10 )
    {
      v11 = v6 + 1;
      if ( v10 < v11 )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v11, 0x10u, a1 + 16, v4);
        v9 = (_QWORD *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
      }
      *v9 = 0;
      v9[1] = v5;
      v5 = v12[0];
      ++*(_DWORD *)(a1 + 8);
    }
    else
    {
      if ( v9 )
      {
        *(_DWORD *)v9 = 0;
        v9[1] = v5;
        v5 = v12[0];
        v8 = *(_DWORD *)(a1 + 8);
      }
      *(_DWORD *)(a1 + 8) = v8 + 1;
    }
  }
  else
  {
    sub_93FB40(a1, 0);
    v5 = v12[0];
  }
  if ( v5 )
LABEL_8:
    sub_B91220((__int64)v12, v5);
}
