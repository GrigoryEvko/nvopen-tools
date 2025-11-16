// Function: sub_39B9D00
// Address: 0x39b9d00
//
__int64 __fastcall sub_39B9D00(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 (*v3)(); // rax
  __int64 v5; // rax
  int v6; // ecx
  int v7; // r9d
  __int64 v8; // r8
  unsigned __int64 v9; // rdx
  unsigned int v10; // r14d
  unsigned int v11; // eax
  __int64 v12; // r14
  __int64 result; // rax
  unsigned __int16 *v14; // r14
  unsigned __int16 v15; // r13
  _QWORD *v16; // rdi
  char v17; // r8
  int v18; // r15d
  char v19; // al
  unsigned __int64 v20; // r15
  __int64 v21; // rax
  unsigned int v22; // r8d
  int v23; // ecx
  unsigned int v24; // r9d
  int v25; // ecx
  __int64 v26; // rdx
  unsigned __int64 v27; // rdx
  unsigned int v28; // r15d
  unsigned int v29; // eax
  unsigned __int64 v30; // rdx
  unsigned int v31; // r15d
  int v32; // r14d
  unsigned __int8 v33; // dl
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rdx
  __int64 v36; // [rsp+0h] [rbp-40h]
  unsigned int v37; // [rsp+0h] [rbp-40h]
  unsigned int v38; // [rsp+8h] [rbp-38h]
  unsigned int v39; // [rsp+8h] [rbp-38h]

  v3 = *(__int64 (**)())(*(_QWORD *)a2[2] + 112LL);
  if ( v3 == sub_1D00B10 )
    BUG();
  v5 = v3();
  v8 = *(_QWORD *)(a3 + 8);
  v9 = *(unsigned int *)(v5 + 16);
  v10 = *(_DWORD *)(v5 + 16);
  if ( v9 <= v8 << 6 )
  {
LABEL_3:
    v11 = *(_DWORD *)(a3 + 16);
    if ( v10 <= v11 )
      goto LABEL_4;
    goto LABEL_29;
  }
  v36 = *(_QWORD *)(a3 + 8);
  v20 = (unsigned int)(v9 + 63) >> 6;
  if ( v20 < 2 * v8 )
    v20 = 2 * v8;
  v21 = (__int64)realloc(*(_QWORD *)a3, 8 * v20, 8 * (int)v20, v6, v8, v7);
  v22 = v36;
  if ( !v21 )
  {
    if ( 8 * v20 )
    {
      v39 = v36;
    }
    else
    {
      v39 = v36;
      v21 = malloc(1u);
      v22 = v36;
      if ( v21 )
        goto LABEL_23;
    }
    sub_16BD1C0("Allocation failed", 1u);
    v22 = v39;
    v21 = 0;
  }
LABEL_23:
  v23 = *(_DWORD *)(a3 + 16);
  *(_QWORD *)a3 = v21;
  *(_QWORD *)(a3 + 8) = v20;
  v24 = (unsigned int)(v23 + 63) >> 6;
  if ( v24 < v20 )
  {
    v37 = (unsigned int)(v23 + 63) >> 6;
    v38 = v22;
    memset((void *)(v21 + 8LL * v24), 0, 8 * (v20 - v24));
    v23 = *(_DWORD *)(a3 + 16);
    v21 = *(_QWORD *)a3;
    v22 = v38;
    v24 = v37;
  }
  v25 = v23 & 0x3F;
  if ( v25 )
  {
    *(_QWORD *)(v21 + 8LL * (v24 - 1)) &= ~(-1LL << v25);
    v21 = *(_QWORD *)a3;
  }
  v26 = *(_QWORD *)(a3 + 8) - v22;
  if ( !v26 )
    goto LABEL_3;
  memset((void *)(v21 + 8LL * v22), 0, 8 * v26);
  v11 = *(_DWORD *)(a3 + 16);
  if ( v10 <= v11 )
    goto LABEL_4;
LABEL_29:
  v27 = *(_QWORD *)(a3 + 8);
  v28 = (v11 + 63) >> 6;
  if ( v27 > v28 )
  {
    v34 = v27 - v28;
    if ( v34 )
    {
      memset((void *)(*(_QWORD *)a3 + 8LL * v28), 0, 8 * v34);
      v11 = *(_DWORD *)(a3 + 16);
    }
  }
  if ( (v11 & 0x3F) == 0 )
  {
LABEL_4:
    *(_DWORD *)(a3 + 16) = v10;
    if ( v10 >= v11 )
      goto LABEL_5;
    goto LABEL_32;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8LL * (v28 - 1)) &= ~(-1LL << (v11 & 0x3F));
  v29 = *(_DWORD *)(a3 + 16);
  *(_DWORD *)(a3 + 16) = v10;
  if ( v10 >= v29 )
    goto LABEL_5;
LABEL_32:
  v30 = *(_QWORD *)(a3 + 8);
  v31 = (v10 + 63) >> 6;
  if ( v30 > v31 )
  {
    v35 = v30 - v31;
    if ( v35 )
    {
      memset((void *)(*(_QWORD *)a3 + 8LL * v31), 0, 8 * v35);
      v10 = *(_DWORD *)(a3 + 16);
    }
  }
  v32 = v10 & 0x3F;
  if ( v32 )
    *(_QWORD *)(*(_QWORD *)a3 + 8LL * (v31 - 1)) &= ~(-1LL << v32);
LABEL_5:
  if ( (*(_BYTE *)(a2[1] + 809) & 1) != 0 )
  {
    v12 = *a2;
    if ( (*(_BYTE *)(*a2 + 32) & 0xFu) - 7 <= 1 && !(unsigned __int8)sub_15E3650(*a2, 0) )
    {
      result = sub_1560180(v12 + 112, 27);
      if ( (_BYTE)result )
      {
        while ( 1 )
        {
          while ( 1 )
          {
            do
            {
              v12 = *(_QWORD *)(v12 + 8);
              if ( !v12 )
                return result;
              result = (__int64)sub_1648700(v12);
              v33 = *(_BYTE *)(result + 16);
            }
            while ( v33 <= 0x17u );
            if ( v33 != 78 )
              break;
            result |= 4uLL;
LABEL_41:
            if ( (result & 0xFFFFFFFFFFFFFFF8LL) != 0 && (result & 4) != 0 )
            {
              result = (*(unsigned __int16 *)((result & 0xFFFFFFFFFFFFFFF8LL) + 18) & 3u) - 1;
              if ( (unsigned int)result <= 1 )
                goto LABEL_8;
            }
          }
          if ( v33 == 29 )
          {
            result &= ~4uLL;
            goto LABEL_41;
          }
        }
      }
    }
  }
LABEL_8:
  result = sub_1E6A620((_QWORD *)a2[5]);
  v14 = (unsigned __int16 *)result;
  if ( result )
  {
    if ( *(_WORD *)result )
    {
      result = sub_1560180(*a2 + 112, 18);
      if ( !(_BYTE)result )
      {
        result = sub_1560180(*a2 + 112, 29);
        if ( !(_BYTE)result
          || (result = sub_1560180(*a2 + 112, 30), !(_BYTE)result)
          || (result = sub_1560180(*a2 + 112, 56), (_BYTE)result)
          || (result = *(_QWORD *)(*(_QWORD *)a1 + 72LL), (__int64 (*)())result == sub_39B9C60)
          || (result = ((__int64 (__fastcall *)(__int64, __int64 *))result)(a1, a2), !(_BYTE)result) )
        {
          v15 = *v14;
          v16 = (_QWORD *)a2[5];
          v17 = *((_BYTE *)a2 + 521);
          if ( *v14 )
          {
            v18 = 0;
            do
            {
              if ( v17 || (v19 = sub_1E6A350(v16, v15, 0), v17 = 0, v19) )
                *(_QWORD *)(*(_QWORD *)a3 + 8LL * (v15 >> 6)) |= 1LL << v15;
              result = (unsigned int)(v18 + 1);
              v15 = v14[result];
              ++v18;
            }
            while ( v15 );
          }
        }
      }
    }
  }
  return result;
}
