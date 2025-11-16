// Function: sub_A88F30
// Address: 0xa88f30
//
void __fastcall sub_A88F30(__int64 a1, __int64 a2, __int64 a3, __int16 a4)
{
  __int64 v5; // rdi
  __int64 v6; // rsi
  __int64 v7; // r13
  unsigned __int64 v8; // rsi
  _QWORD *v9; // rax
  int v10; // ecx
  _QWORD *v11; // rdx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rsi
  _QWORD v14[5]; // [rsp+18h] [rbp-28h] BYREF

  *(_QWORD *)(a1 + 48) = a2;
  *(_QWORD *)(a1 + 56) = a3;
  *(_WORD *)(a1 + 64) = a4;
  if ( a3 != a2 + 48 )
  {
    v5 = a3 - 24;
    if ( !a3 )
      v5 = 0;
    v6 = *(_QWORD *)sub_B46C60(v5);
    v14[0] = v6;
    if ( v6 && (sub_B96E90(v14, v6, 1), (v7 = v14[0]) != 0) )
    {
      v8 = *(unsigned int *)(a1 + 8);
      v9 = *(_QWORD **)a1;
      v10 = *(_DWORD *)(a1 + 8);
      v11 = (_QWORD *)(*(_QWORD *)a1 + 16 * v8);
      if ( *(_QWORD **)a1 != v11 )
      {
        while ( *(_DWORD *)v9 )
        {
          v9 += 2;
          if ( v11 == v9 )
            goto LABEL_16;
        }
        v9[1] = v14[0];
        goto LABEL_11;
      }
LABEL_16:
      v12 = *(unsigned int *)(a1 + 12);
      if ( v8 >= v12 )
      {
        v13 = v8 + 1;
        if ( v12 < v13 )
        {
          sub_C8D5F0(a1, a1 + 16, v13, 16);
          v11 = (_QWORD *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
        }
        *v11 = 0;
        v11[1] = v7;
        v7 = v14[0];
        ++*(_DWORD *)(a1 + 8);
      }
      else
      {
        if ( v11 )
        {
          *(_DWORD *)v11 = 0;
          v11[1] = v7;
          v7 = v14[0];
          v10 = *(_DWORD *)(a1 + 8);
        }
        *(_DWORD *)(a1 + 8) = v10 + 1;
      }
    }
    else
    {
      sub_93FB40(a1, 0);
      v7 = v14[0];
    }
    if ( v7 )
LABEL_11:
      sub_B91220(v14);
  }
}
