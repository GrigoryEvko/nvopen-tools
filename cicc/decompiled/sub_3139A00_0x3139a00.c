// Function: sub_3139A00
// Address: 0x3139a00
//
void __fastcall sub_3139A00(__int64 a1, __int64 a2, __int64 a3, char a4, char a5)
{
  __int16 v5; // ax
  __int64 v6; // rsi
  __int64 v7; // r9
  __int64 v8; // r13
  unsigned __int64 v9; // rsi
  _QWORD *v10; // rax
  int v11; // ecx
  _QWORD *v12; // rdx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rsi
  _QWORD v15[5]; // [rsp+18h] [rbp-28h] BYREF

  if ( !a2 )
  {
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = 0;
    *(_WORD *)(a1 + 64) = 0;
    return;
  }
  *(_QWORD *)(a1 + 48) = a2;
  LOBYTE(v5) = a4;
  HIBYTE(v5) = a5;
  *(_QWORD *)(a1 + 56) = a3;
  *(_WORD *)(a1 + 64) = v5;
  if ( a3 != a2 + 48 )
  {
    if ( a3 )
      a3 -= 24;
    v6 = *(_QWORD *)sub_B46C60(a3);
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
            goto LABEL_18;
        }
        v10[1] = v15[0];
        goto LABEL_12;
      }
LABEL_18:
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
LABEL_12:
      sub_B91220((__int64)v15, v8);
  }
}
