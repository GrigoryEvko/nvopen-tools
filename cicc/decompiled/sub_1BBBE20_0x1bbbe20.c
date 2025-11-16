// Function: sub_1BBBE20
// Address: 0x1bbbe20
//
__int64 __fastcall sub_1BBBE20(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 result; // rax
  __int64 *v7; // r10
  __int64 *v8; // r13
  __int64 *v11; // r15
  __int64 v12; // r10
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // r14
  unsigned __int64 v16; // r13
  int v17; // r14d
  unsigned __int64 v18; // r11
  __int64 v19; // rdi
  __int64 v20; // r15
  __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // r9
  unsigned int v24; // ecx
  char v25; // al
  unsigned int v26; // ecx
  __int64 v27; // r9
  __int64 v28; // rsi
  unsigned int v29; // ecx
  char v30; // al
  unsigned int v31; // ecx
  __int64 v32; // r9
  __int64 *v33; // r9
  __int64 v34; // rdx
  __int64 *v35; // r15
  __int64 v36; // rdx
  char v37; // al
  __int64 *v38; // [rsp+8h] [rbp-68h]
  __int64 *v39; // [rsp+8h] [rbp-68h]
  unsigned __int64 v40; // [rsp+10h] [rbp-60h]
  unsigned __int64 v41; // [rsp+10h] [rbp-60h]
  __int64 v43; // [rsp+20h] [rbp-50h]
  __int64 *v44; // [rsp+28h] [rbp-48h]
  __int64 *v45; // [rsp+28h] [rbp-48h]
  const void *v46; // [rsp+30h] [rbp-40h]
  unsigned __int64 v47; // [rsp+30h] [rbp-40h]
  unsigned __int64 v48; // [rsp+30h] [rbp-40h]
  __int64 v50; // [rsp+38h] [rbp-38h]
  __int64 v51; // [rsp+38h] [rbp-38h]
  __int64 v52; // [rsp+38h] [rbp-38h]
  __int64 v53; // [rsp+38h] [rbp-38h]

  result = a4 + 16;
  v7 = a2;
  v8 = &a2[a3];
  v46 = (const void *)(a4 + 16);
  if ( a2 != v8 )
  {
    v11 = a2;
    while ( 1 )
    {
      v15 = *v11;
      if ( (*(_BYTE *)(*v11 + 23) & 0x40) != 0 )
      {
        v12 = **(_QWORD **)(v15 - 8);
        v13 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v13 < *(_DWORD *)(a4 + 12) )
          goto LABEL_4;
      }
      else
      {
        v12 = *(_QWORD *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
        v13 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v13 < *(_DWORD *)(a4 + 12) )
          goto LABEL_4;
      }
      v43 = v12;
      sub_16CD150(a4, v46, 0, 8, a5, a6);
      v13 = *(unsigned int *)(a4 + 8);
      v12 = v43;
LABEL_4:
      *(_QWORD *)(*(_QWORD *)a4 + 8 * v13) = v12;
      ++*(_DWORD *)(a4 + 8);
      if ( (*(_BYTE *)(v15 + 23) & 0x40) != 0 )
      {
        result = *(unsigned int *)(a5 + 8);
        v14 = *(_QWORD *)(*(_QWORD *)(v15 - 8) + 24LL);
        if ( (unsigned int)result >= *(_DWORD *)(a5 + 12) )
          goto LABEL_11;
      }
      else
      {
        result = *(unsigned int *)(a5 + 8);
        v14 = *(_QWORD *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF) + 24);
        if ( (unsigned int)result >= *(_DWORD *)(a5 + 12) )
        {
LABEL_11:
          sub_16CD150(a5, (const void *)(a5 + 16), 0, 8, a5, a6);
          result = *(unsigned int *)(a5 + 8);
        }
      }
      ++v11;
      *(_QWORD *)(*(_QWORD *)a5 + 8 * result) = v14;
      ++*(_DWORD *)(a5 + 8);
      if ( v8 == v11 )
      {
        v7 = a2;
        break;
      }
    }
  }
  v16 = 0;
  v17 = 0;
  v18 = a3 - 1;
  if ( a3 != 1 )
  {
    do
    {
      v20 = 8 * v16;
      v21 = *(_QWORD *)(*(_QWORD *)a4 + 8 * v16);
      v16 = (unsigned int)(v17 + 1);
      result = *(_QWORD *)a5;
      ++v17;
      if ( *(_BYTE *)(v21 + 16) == 54
        && (v22 = *(_QWORD *)(result + 8 * v16), v23 = 8 * v16, *(_BYTE *)(v22 + 16) == 54) )
      {
        v50 = v7[v16];
        v24 = *(unsigned __int8 *)(v7[(unsigned __int64)v20 / 8] + 16) - 24;
        if ( v24 <= 0x1C && ((1LL << v24) & 0x1C019800) != 0 )
        {
          v38 = v7;
          v40 = v18;
          v25 = sub_385F290(v21, v22, *(_QWORD *)(a1 + 1376), *(_QWORD *)(a1 + 1312), 1);
          v23 = 8 * v16;
          v18 = v40;
          v7 = v38;
          if ( v25 )
          {
LABEL_34:
            result = v20 + *(_QWORD *)a5;
            v35 = (__int64 *)(*(_QWORD *)a4 + v20);
            v36 = *v35;
            *v35 = *(_QWORD *)result;
            *(_QWORD *)result = v36;
            continue;
          }
        }
        v26 = *(unsigned __int8 *)(v50 + 16) - 24;
        if ( v26 <= 0x1C && ((1LL << v26) & 0x1C019800) != 0 )
        {
          v45 = v7;
          v48 = v18;
          v53 = v23;
          v37 = sub_385F290(v21, v22, *(_QWORD *)(a1 + 1376), *(_QWORD *)(a1 + 1312), 1);
          v32 = v53;
          v18 = v48;
          v7 = v45;
          if ( v37 )
            goto LABEL_32;
        }
        result = *(_QWORD *)a5;
        v19 = *(_QWORD *)(*(_QWORD *)a5 + v20);
        if ( *(_BYTE *)(v19 + 16) != 54 )
          continue;
      }
      else
      {
        v19 = *(_QWORD *)(result + v20);
        if ( *(_BYTE *)(v19 + 16) != 54 )
          continue;
      }
      result = *(_QWORD *)a4;
      v27 = 8 * v16;
      v28 = *(_QWORD *)(*(_QWORD *)a4 + 8 * v16);
      if ( *(_BYTE *)(v28 + 16) == 54 )
      {
        v51 = v7[v16];
        v29 = *(unsigned __int8 *)(v7[(unsigned __int64)v20 / 8] + 16) - 24;
        if ( v29 <= 0x1C && ((1LL << v29) & 0x1C019800) != 0 )
        {
          v39 = v7;
          v41 = v18;
          v30 = sub_385F290(v19, v28, *(_QWORD *)(a1 + 1376), *(_QWORD *)(a1 + 1312), 1);
          v27 = 8 * v16;
          v18 = v41;
          v7 = v39;
          if ( v30 )
            goto LABEL_34;
        }
        result = v51;
        v31 = *(unsigned __int8 *)(v51 + 16) - 24;
        if ( v31 > 0x1C )
          continue;
        result = 1LL << v31;
        if ( ((1LL << v31) & 0x1C019800) == 0 )
          continue;
        v44 = v7;
        v47 = v18;
        v52 = v27;
        result = sub_385F290(v19, v28, *(_QWORD *)(a1 + 1376), *(_QWORD *)(a1 + 1312), 1);
        v32 = v52;
        v18 = v47;
        v7 = v44;
        if ( !(_BYTE)result )
          continue;
LABEL_32:
        result = v32 + *(_QWORD *)a5;
        v33 = (__int64 *)(*(_QWORD *)a4 + v32);
        v34 = *v33;
        *v33 = *(_QWORD *)result;
        *(_QWORD *)result = v34;
      }
    }
    while ( v18 > v16 );
  }
  return result;
}
