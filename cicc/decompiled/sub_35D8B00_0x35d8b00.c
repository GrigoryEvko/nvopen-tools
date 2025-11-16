// Function: sub_35D8B00
// Address: 0x35d8b00
//
__int64 __fastcall sub_35D8B00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r8
  unsigned int v6; // r14d
  __int64 v7; // r11
  unsigned __int64 v8; // rsi
  unsigned int v9; // eax
  __int64 v10; // r13
  unsigned int v11; // ebx
  unsigned __int64 v12; // rdx
  unsigned int v13; // eax
  unsigned __int64 v14; // r12
  bool v15; // cc
  unsigned __int64 v16; // rdx
  int v17; // ecx
  __int64 v18; // r15
  __int64 v19; // r15
  unsigned __int64 v20; // rax
  unsigned int i; // edx
  unsigned int v22; // eax
  unsigned int v23; // ecx
  unsigned int v24; // eax
  unsigned int v26; // eax
  __int64 v27; // [rsp+0h] [rbp-50h]
  __int64 v28; // [rsp+0h] [rbp-50h]
  __int64 v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+8h] [rbp-48h]
  unsigned int v31; // [rsp+1Ch] [rbp-34h]
  unsigned int v32; // [rsp+1Ch] [rbp-34h]

  v5 = a3;
  v6 = 0x80000000;
  v7 = *(_QWORD *)(a3 + 8);
  v8 = *(_QWORD *)(a3 + 16);
  v9 = *(_DWORD *)(a3 + 40) >> 1;
  v10 = v7;
  v11 = v9 + *(_DWORD *)(v7 + 32);
  if ( v9 + (unsigned __int64)*(unsigned int *)(v7 + 32) > 0x80000000 )
    v11 = 0x80000000;
  v12 = v9 + (unsigned __int64)*(unsigned int *)(v8 + 32);
  v13 = *(_DWORD *)(v8 + 32) + v9;
  v14 = v8;
  v15 = v12 <= 0x80000000;
  v16 = v7 + 40;
  if ( v15 )
    v6 = v13;
  v17 = 0;
  if ( v8 <= v16 )
    goto LABEL_21;
  do
  {
    while ( 1 )
    {
      if ( v6 <= v11 && (v6 != v11 || (v17 & 1) == 0) )
      {
        v18 = *(unsigned int *)(v14 - 8);
        if ( v18 + (unsigned __int64)v6 > 0x80000000 )
        {
          v14 -= 40LL;
          v6 = 0x80000000;
        }
        else
        {
          v6 += v18;
          v14 -= 40LL;
        }
        goto LABEL_10;
      }
      v19 = *(unsigned int *)(v10 + 72);
      if ( v19 + (unsigned __int64)v11 <= 0x80000000 )
        break;
      v10 = v16;
      v11 = 0x80000000;
LABEL_10:
      v16 = v10 + 40;
      ++v17;
      if ( v14 <= v10 + 40 )
        goto LABEL_14;
    }
    v10 = v16;
    v11 += v19;
    ++v17;
    v16 += 40LL;
  }
  while ( v14 > v16 );
LABEL_14:
  v20 = v8;
  for ( i = -858993459 * ((v10 - v7) >> 3) + 1; ; i = -858993459 * ((v10 - *(_QWORD *)(v5 + 8)) >> 3) + 1 )
  {
    v22 = -858993459 * ((__int64)(v20 - v14) >> 3) + 1;
    v23 = v22;
    if ( i <= v22 )
      v23 = i;
    if ( v23 > 2 )
      break;
    if ( v22 > i )
    {
      if ( v22 <= 3 )
        break;
      v28 = a1;
      v30 = v5;
      v32 = sub_35D8A60(a2, v14, v14, *(_QWORD *)(v5 + 16));
      v26 = sub_35D8A60(a2, v14, *(_QWORD *)(v30 + 8), v10);
      v5 = v30;
      a1 = v28;
      if ( v32 < v26 )
        break;
      v10 += 40;
      v14 += 40LL;
    }
    else
    {
      if ( i <= 3 )
        break;
      v27 = a1;
      v29 = v5;
      v31 = sub_35D8A60(a2, v10, *(_QWORD *)(v5 + 8), v10);
      v24 = sub_35D8A60(a2, v10, v14, *(_QWORD *)(v29 + 16));
      v5 = v29;
      a1 = v27;
      if ( v31 < v24 )
        break;
      v10 -= 40;
      v14 -= 40LL;
    }
    v20 = *(_QWORD *)(v5 + 16);
  }
LABEL_21:
  *(_QWORD *)a1 = v10;
  *(_QWORD *)(a1 + 8) = v14;
  *(_DWORD *)(a1 + 16) = v11;
  *(_DWORD *)(a1 + 20) = v6;
  return a1;
}
