// Function: sub_2412230
// Address: 0x2412230
//
void __fastcall sub_2412230(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int16 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r14
  unsigned __int64 v17; // rsi
  _QWORD *v18; // rax
  int v19; // ecx
  _QWORD *v20; // rdx
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rsi
  _QWORD v24[7]; // [rsp+38h] [rbp-38h] BYREF

  v11 = sub_AA48A0(a2);
  *(_QWORD *)(a1 + 48) = a2;
  *(_QWORD *)(a1 + 72) = v11;
  *(_QWORD *)(a1 + 80) = a1 + 128;
  *(_QWORD *)(a1 + 88) = a1 + 136;
  *(_QWORD *)(a1 + 96) = a5;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 128) = &unk_49DA100;
  *(_QWORD *)(a1 + 8) = 0x200000000LL;
  *(_WORD *)(a1 + 108) = 512;
  *(_DWORD *)(a1 + 104) = 0;
  *(_BYTE *)(a1 + 110) = 7;
  *(_QWORD *)(a1 + 112) = a7;
  *(_QWORD *)(a1 + 120) = a8;
  *(_QWORD *)(a1 + 136) = &unk_49DA0B0;
  *(_QWORD *)(a1 + 56) = a3;
  *(_WORD *)(a1 + 64) = a4;
  if ( a3 != a2 + 48 )
  {
    v12 = a3 - 24;
    if ( !a3 )
      v12 = 0;
    v13 = *(_QWORD *)sub_B46C60(v12);
    v24[0] = v13;
    if ( v13 && (sub_B96E90((__int64)v24, v13, 1), (v16 = v24[0]) != 0) )
    {
      v17 = *(unsigned int *)(a1 + 8);
      v18 = *(_QWORD **)a1;
      v19 = *(_DWORD *)(a1 + 8);
      v20 = (_QWORD *)(*(_QWORD *)a1 + 16 * v17);
      if ( *(_QWORD **)a1 != v20 )
      {
        while ( *(_DWORD *)v18 )
        {
          v18 += 2;
          if ( v20 == v18 )
            goto LABEL_16;
        }
        v18[1] = v24[0];
        goto LABEL_11;
      }
LABEL_16:
      v21 = *(unsigned int *)(a1 + 12);
      if ( v17 >= v21 )
      {
        v22 = v17 + 1;
        if ( v21 < v22 )
        {
          sub_C8D5F0(a1, (const void *)(a1 + 16), v22, 0x10u, v14, v15);
          v20 = (_QWORD *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
        }
        *v20 = 0;
        v20[1] = v16;
        v16 = v24[0];
        ++*(_DWORD *)(a1 + 8);
      }
      else
      {
        if ( v20 )
        {
          *(_DWORD *)v20 = 0;
          v20[1] = v16;
          v16 = v24[0];
          v19 = *(_DWORD *)(a1 + 8);
        }
        *(_DWORD *)(a1 + 8) = v19 + 1;
      }
    }
    else
    {
      sub_93FB40(a1, 0);
      v16 = v24[0];
    }
    if ( v16 )
LABEL_11:
      sub_B91220((__int64)v24, v16);
  }
}
