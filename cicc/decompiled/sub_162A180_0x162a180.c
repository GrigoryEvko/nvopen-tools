// Function: sub_162A180
// Address: 0x162a180
//
__int64 __fastcall sub_162A180(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  int v4; // r14d
  __int64 v5; // r13
  char v6; // al
  _QWORD *v7; // rdx
  __int64 result; // rax
  int v9; // esi
  int v10; // eax
  unsigned int v11; // esi
  __int64 *v12; // rdx
  __int64 v13; // rcx
  int i; // r10d
  unsigned int v15; // esi
  int v16; // eax
  int v17; // eax
  unsigned __int64 v18; // r11
  int v19; // ecx
  __int64 v20; // rcx
  unsigned __int64 v21; // rcx
  unsigned int v22; // r15d
  __int64 *v23; // r14
  __int64 v24; // r15
  unsigned int v25; // ecx
  __int64 *v26; // r15
  __int64 v27; // r15
  __int64 v28; // [rsp+0h] [rbp-50h]
  __int64 v29; // [rsp+8h] [rbp-48h] BYREF
  _QWORD *v30; // [rsp+10h] [rbp-40h] BYREF
  __int64 v31; // [rsp+18h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 8);
  v4 = *(_DWORD *)(a2 + 24);
  v29 = a1;
  v5 = *(_QWORD *)(a2 + 8);
  v30 = *(_QWORD **)(a1 - 8 * v3);
  v31 = *(_QWORD *)(a1 + 24);
  if ( !v4 )
    goto LABEL_2;
  v9 = sub_1620E10((__int64 *)&v30);
  v10 = v4 - 1;
  v11 = (v4 - 1) & v9;
  v12 = (__int64 *)(v5 + 8LL * v11);
  v13 = *v12;
  if ( *v12 == -8 )
    goto LABEL_2;
  for ( i = 1; ; ++i )
  {
    if ( v13 == -16 || v31 != *(_QWORD *)(v13 + 24) )
      goto LABEL_8;
    v18 = *(_QWORD *)(v13 - 8LL * *(unsigned int *)(v13 + 8));
    v19 = *(unsigned __int8 *)v18;
    if ( (_BYTE)v19 == 1 )
    {
      v20 = *(_QWORD *)(v18 + 136);
    }
    else
    {
      if ( (unsigned int)(v19 - 24) > 1 )
        goto LABEL_25;
      v20 = v18 | 4;
    }
    if ( (v20 & 4) == 0 )
    {
      v21 = v20 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v21 )
      {
        if ( *(_BYTE *)v30 == 1 )
        {
          v22 = *(_DWORD *)(v21 + 32);
          v23 = *(__int64 **)(v21 + 24);
          v28 = v22 > 0x40 ? *v23 : (__int64)((_QWORD)v23 << (64 - (unsigned __int8)v22)) >> (64 - (unsigned __int8)v22);
          v24 = v30[17];
          v25 = *(_DWORD *)(v24 + 32);
          v26 = *(__int64 **)(v24 + 24);
          v27 = v25 > 0x40 ? *v26 : (__int64)((_QWORD)v26 << (64 - (unsigned __int8)v25)) >> (64 - (unsigned __int8)v25);
          if ( v27 == v28 )
            break;
        }
      }
    }
LABEL_25:
    if ( v30 == (_QWORD *)v18 )
      break;
LABEL_8:
    v11 = v10 & (i + v11);
    v12 = (__int64 *)(v5 + 8LL * v11);
    v13 = *v12;
    if ( *v12 == -8 )
      goto LABEL_2;
  }
  if ( v12 == (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) || (result = *v12) == 0 )
  {
LABEL_2:
    v6 = sub_15B7360(a2, &v29, &v30);
    v7 = v30;
    if ( v6 )
      return v29;
    v15 = *(_DWORD *)(a2 + 24);
    v16 = *(_DWORD *)(a2 + 16);
    ++*(_QWORD *)a2;
    v17 = v16 + 1;
    if ( 4 * v17 >= 3 * v15 )
    {
      v15 *= 2;
    }
    else if ( v15 - *(_DWORD *)(a2 + 20) - v17 > v15 >> 3 )
    {
LABEL_12:
      *(_DWORD *)(a2 + 16) = v17;
      if ( *v7 != -8 )
        --*(_DWORD *)(a2 + 20);
      *v7 = v29;
      return v29;
    }
    sub_15BAB20(a2, v15);
    sub_15B7360(a2, &v29, &v30);
    v7 = v30;
    v17 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_12;
  }
  return result;
}
