// Function: sub_30E1100
// Address: 0x30e1100
//
__int64 __fastcall sub_30E1100(__int64 a1, __int64 a2, __int64 a3, int *a4)
{
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // rdx
  int v12; // r11d
  unsigned int i; // eax
  __int64 v14; // r9
  unsigned int v15; // eax
  __int64 v16; // rbx
  __int64 *v17; // rax
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r15
  __int64 *v24; // [rsp+8h] [rbp-E8h]
  __int64 v26; // [rsp+18h] [rbp-D8h]
  __int64 v27; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v28; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v29; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v30; // [rsp+38h] [rbp-B8h]
  __int64 v31; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v32; // [rsp+48h] [rbp-A8h]
  char v33; // [rsp+C0h] [rbp-30h] BYREF

  v7 = sub_B491C0(a2);
  v8 = *(_QWORD *)(sub_BC1CD0(a3, &unk_4F82410, v7) + 8);
  v9 = *(_QWORD *)(v8 + 72);
  v10 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL) + 40LL);
  v11 = *(unsigned int *)(v8 + 88);
  if ( !(_DWORD)v11 )
    goto LABEL_22;
  v12 = 1;
  for ( i = (v11 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F87C68 >> 9) ^ ((unsigned int)&unk_4F87C68 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)))); ; i = (v11 - 1) & v15 )
  {
    v14 = v9 + 24LL * i;
    if ( *(_UNKNOWN **)v14 == &unk_4F87C68 && v10 == *(_QWORD *)(v14 + 8) )
      break;
    if ( *(_QWORD *)v14 == -4096 && *(_QWORD *)(v14 + 8) == -4096 )
      goto LABEL_22;
    v15 = v12 + i;
    ++v12;
  }
  if ( v14 == v9 + 24 * v11 )
  {
LABEL_22:
    v16 = 0;
  }
  else
  {
    v16 = *(_QWORD *)(*(_QWORD *)(v14 + 16) + 24LL);
    if ( v16 )
    {
      v30 = 1;
      v16 += 8;
      v17 = &v31;
      do
      {
        *v17 = -4096;
        v17 += 2;
      }
      while ( v17 != (__int64 *)&v33 );
      if ( (v30 & 1) == 0 )
        sub_C7D6A0(v31, 16LL * v32, 8);
    }
  }
  v18 = sub_BC1CD0(a3, &unk_4F8FAE8, v7);
  v19 = *(_QWORD *)(a2 - 32);
  v27 = a3;
  v26 = v18;
  v28 = a3;
  v29 = a3;
  if ( v19 )
  {
    if ( *(_BYTE *)v19 )
    {
      v19 = 0;
    }
    else if ( *(_QWORD *)(v19 + 24) != *(_QWORD *)(a2 + 80) )
    {
      v19 = 0;
    }
  }
  v24 = (__int64 *)(sub_BC1CD0(a3, &unk_4F89C30, v19) + 8);
  v20 = sub_B2BE50(v19);
  v21 = sub_B6F970(v20);
  v22 = v26 + 8;
  if ( !(*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v21 + 32LL))(
          v21,
          "inline-order",
          12) )
    v22 = 0;
  sub_30DF350(
    a1,
    a2,
    a4,
    v24,
    (__int64)sub_30E0DE0,
    (__int64)&v27,
    (__int64)sub_30E0DC0,
    (__int64)&v29,
    (__int64)sub_30E0E00,
    (__int64)&v28,
    v16,
    v22);
  return a1;
}
