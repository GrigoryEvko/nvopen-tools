// Function: sub_25B0770
// Address: 0x25b0770
//
void __fastcall sub_25B0770(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r14
  unsigned __int64 v12; // rsi
  _QWORD *v13; // rax
  int v14; // ecx
  _QWORD *v15; // rdx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rsi
  _QWORD v19[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = sub_BD5C60(a2);
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 72) = v7;
  *(_QWORD *)(a1 + 80) = a1 + 128;
  *(_QWORD *)(a1 + 88) = a1 + 136;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = 0x200000000LL;
  *(_QWORD *)(a1 + 128) = &unk_49DA1B0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 96) = a3;
  *(_DWORD *)(a1 + 104) = 0;
  *(_WORD *)(a1 + 108) = 512;
  *(_BYTE *)(a1 + 110) = 7;
  *(_QWORD *)(a1 + 112) = a4;
  *(_QWORD *)(a1 + 120) = a5;
  *(_WORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 136) = &unk_49DA0B0;
  *(_QWORD *)(a1 + 48) = *(_QWORD *)(a2 + 40);
  *(_QWORD *)(a1 + 56) = a2 + 24;
  v8 = *(_QWORD *)sub_B46C60(a2);
  v19[0] = v8;
  if ( v8 && (sub_B96E90((__int64)v19, v8, 1), (v11 = v19[0]) != 0) )
  {
    v12 = *(unsigned int *)(a1 + 8);
    v13 = *(_QWORD **)a1;
    v14 = *(_DWORD *)(a1 + 8);
    v15 = (_QWORD *)(*(_QWORD *)a1 + 16 * v12);
    if ( *(_QWORD **)a1 != v15 )
    {
      while ( *(_DWORD *)v13 )
      {
        v13 += 2;
        if ( v15 == v13 )
          goto LABEL_13;
      }
      v13[1] = v19[0];
      goto LABEL_8;
    }
LABEL_13:
    v16 = *(unsigned int *)(a1 + 12);
    if ( v12 >= v16 )
    {
      v17 = v12 + 1;
      if ( v16 < v17 )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v17, 0x10u, v9, v10);
        v15 = (_QWORD *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
      }
      *v15 = 0;
      v15[1] = v11;
      v11 = v19[0];
      ++*(_DWORD *)(a1 + 8);
    }
    else
    {
      if ( v15 )
      {
        *(_DWORD *)v15 = 0;
        v15[1] = v11;
        v11 = v19[0];
        v14 = *(_DWORD *)(a1 + 8);
      }
      *(_DWORD *)(a1 + 8) = v14 + 1;
    }
  }
  else
  {
    sub_93FB40(a1, 0);
    v11 = v19[0];
  }
  if ( v11 )
LABEL_8:
    sub_B91220((__int64)v19, v11);
}
