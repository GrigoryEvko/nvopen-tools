// Function: sub_24CC710
// Address: 0x24cc710
//
void __fastcall sub_24CC710(__int64 a1, __int64 a2, __int64 a3, __int16 a4)
{
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r14
  unsigned __int64 v14; // rsi
  _QWORD *v15; // rax
  int v16; // ecx
  _QWORD *v17; // rdx
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rsi
  _QWORD v20[7]; // [rsp+18h] [rbp-38h] BYREF

  v8 = sub_AA48A0(a2);
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 72) = v8;
  *(_QWORD *)(a1 + 80) = a1 + 128;
  *(_QWORD *)(a1 + 88) = a1 + 136;
  *(_QWORD *)(a1 + 8) = 0x200000000LL;
  *(_WORD *)(a1 + 108) = 512;
  *(_QWORD *)(a1 + 128) = &unk_49DA100;
  *(_QWORD *)(a1 + 96) = 0;
  *(_BYTE *)(a1 + 110) = 7;
  *(_QWORD *)(a1 + 136) = &unk_49DA0B0;
  *(_DWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 48) = a2;
  *(_QWORD *)(a1 + 56) = a3;
  *(_WORD *)(a1 + 64) = a4;
  if ( a3 != a2 + 48 )
  {
    v9 = a3 - 24;
    if ( !a3 )
      v9 = 0;
    v10 = *(_QWORD *)sub_B46C60(v9);
    v20[0] = v10;
    if ( v10 && (sub_B96E90((__int64)v20, v10, 1), (v13 = v20[0]) != 0) )
    {
      v14 = *(unsigned int *)(a1 + 8);
      v15 = *(_QWORD **)a1;
      v16 = *(_DWORD *)(a1 + 8);
      v17 = (_QWORD *)(*(_QWORD *)a1 + 16 * v14);
      if ( *(_QWORD **)a1 != v17 )
      {
        while ( *(_DWORD *)v15 )
        {
          v15 += 2;
          if ( v17 == v15 )
            goto LABEL_16;
        }
        v15[1] = v20[0];
LABEL_11:
        sub_B91220((__int64)v20, v13);
        goto LABEL_12;
      }
LABEL_16:
      v18 = *(unsigned int *)(a1 + 12);
      if ( v14 >= v18 )
      {
        v19 = v14 + 1;
        if ( v18 < v19 )
        {
          sub_C8D5F0(a1, (const void *)(a1 + 16), v19, 0x10u, v11, v12);
          v17 = (_QWORD *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
        }
        *v17 = 0;
        v17[1] = v13;
        v13 = v20[0];
        ++*(_DWORD *)(a1 + 8);
      }
      else
      {
        if ( v17 )
        {
          *(_DWORD *)v17 = 0;
          v17[1] = v13;
          v13 = v20[0];
          v16 = *(_DWORD *)(a1 + 8);
        }
        *(_DWORD *)(a1 + 8) = v16 + 1;
      }
    }
    else
    {
      sub_93FB40(a1, 0);
      v13 = v20[0];
    }
    if ( !v13 )
      goto LABEL_12;
    goto LABEL_11;
  }
LABEL_12:
  sub_24CC5B0(a1, *(_QWORD *)(a2 + 72));
}
