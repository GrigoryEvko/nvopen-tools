// Function: sub_18FEF10
// Address: 0x18fef10
//
_QWORD *__fastcall sub_18FEF10(__int64 a1, __int64 a2, __int64 *a3, _QWORD *a4)
{
  _QWORD *v4; // r11
  unsigned int v6; // ebx
  int v7; // ebx
  __int64 *v8; // rcx
  __int64 v9; // r12
  int v10; // ebx
  int v11; // eax
  char v12; // al
  int v13; // eax
  __int64 v14; // rax
  __int64 v15; // r12
  _QWORD *result; // rax
  __int64 v17; // rbx
  __int64 v18; // r12
  unsigned int v19; // ebx
  int v20; // eax
  char v21; // al
  __int64 v22; // rdx
  int v23; // eax
  int v24; // ebx
  __int64 v25; // r12
  int v26; // ebx
  int v27; // eax
  char v28; // al
  int v29; // [rsp+Ch] [rbp-54h]
  int v30; // [rsp+Ch] [rbp-54h]
  int v31; // [rsp+Ch] [rbp-54h]
  __int64 *v32; // [rsp+10h] [rbp-50h]
  __int64 *v33; // [rsp+10h] [rbp-50h]
  _QWORD *v34; // [rsp+18h] [rbp-48h]
  _QWORD *v35; // [rsp+18h] [rbp-48h]
  _QWORD *v36; // [rsp+18h] [rbp-48h]
  __int64 *v37; // [rsp+20h] [rbp-40h]
  _QWORD *v38; // [rsp+20h] [rbp-40h]
  __int64 *v40; // [rsp+20h] [rbp-40h]
  __int64 *v41; // [rsp+20h] [rbp-40h]
  _QWORD *v42; // [rsp+28h] [rbp-38h]
  unsigned int v43; // [rsp+28h] [rbp-38h]
  __int64 *v44; // [rsp+28h] [rbp-38h]
  unsigned int v45; // [rsp+28h] [rbp-38h]
  unsigned int v46; // [rsp+28h] [rbp-38h]

  v4 = a4;
  v6 = *(_DWORD *)(a1 + 24);
  if ( v6 )
  {
    v18 = *(_QWORD *)(a1 + 8);
    v19 = v6 - 1;
    v20 = sub_18FDEE0(*a3);
    v30 = 1;
    v4 = a4;
    v33 = 0;
    v45 = v19 & v20;
    while ( 1 )
    {
      v35 = v4;
      v40 = (__int64 *)(v18 + 16LL * v45);
      v21 = sub_18FB980(*a3, *v40);
      v8 = v40;
      v4 = v35;
      if ( v21 )
      {
        result = *(_QWORD **)(a1 + 40);
        v15 = v40[1];
        v17 = *(_QWORD *)(a2 + 16);
        if ( !result )
          goto LABEL_10;
        goto LABEL_14;
      }
      if ( *v40 == -8 )
        break;
      if ( *v40 == -16 )
      {
        if ( *v40 == -8 )
          break;
        if ( !v33 )
        {
          if ( *v40 != -16 )
            v8 = 0;
          v33 = v8;
        }
      }
      v45 = v19 & (v30 + v45);
      ++v30;
    }
    v6 = *(_DWORD *)(a1 + 24);
    if ( v33 )
      v8 = v33;
    v23 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v13 = v23 + 1;
    if ( 4 * v13 >= 3 * v6 )
      goto LABEL_3;
    if ( v6 - (v13 + *(_DWORD *)(a1 + 20)) > v6 >> 3 )
      goto LABEL_7;
    sub_18FE1A0(a1, v6);
    v24 = *(_DWORD *)(a1 + 24);
    v8 = 0;
    v4 = v35;
    if ( v24 )
    {
      v25 = *(_QWORD *)(a1 + 8);
      v26 = v24 - 1;
      v27 = sub_18FDEE0(*a3);
      v31 = 1;
      v4 = v35;
      v32 = 0;
      v46 = v26 & v27;
      while ( 1 )
      {
        v36 = v4;
        v41 = (__int64 *)(v25 + 16LL * v46);
        v28 = sub_18FB980(*a3, *v41);
        v8 = v41;
        v4 = v36;
        if ( v28 )
          break;
        if ( *v41 == -8 )
          goto LABEL_34;
        if ( *v41 == -16 )
        {
          if ( *v41 == -8 )
          {
LABEL_34:
            if ( v32 )
              v8 = v32;
            break;
          }
          if ( !v32 )
          {
            if ( *v41 != -16 )
              v8 = 0;
            v32 = v8;
          }
        }
        v46 = v26 & (v31 + v46);
        ++v31;
      }
    }
  }
  else
  {
    ++*(_QWORD *)a1;
LABEL_3:
    v42 = v4;
    sub_18FE1A0(a1, 2 * v6);
    v7 = *(_DWORD *)(a1 + 24);
    v8 = 0;
    v4 = v42;
    if ( v7 )
    {
      v9 = *(_QWORD *)(a1 + 8);
      v10 = v7 - 1;
      v11 = sub_18FDEE0(*a3);
      v29 = 1;
      v4 = v42;
      v32 = 0;
      v43 = v10 & v11;
      while ( 1 )
      {
        v34 = v4;
        v37 = (__int64 *)(v9 + 16LL * v43);
        v12 = sub_18FB980(*a3, *v37);
        v8 = v37;
        v4 = v34;
        if ( v12 )
          break;
        if ( *v37 == -8 )
          goto LABEL_34;
        if ( *v37 == -16 )
        {
          if ( *v37 == -8 )
            goto LABEL_34;
          if ( !v32 )
          {
            if ( *v37 != -16 )
              v8 = 0;
            v32 = v8;
          }
        }
        v43 = v10 & (v29 + v43);
        ++v29;
      }
    }
  }
  v13 = *(_DWORD *)(a1 + 16) + 1;
LABEL_7:
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v8 != -8 )
    --*(_DWORD *)(a1 + 20);
  v14 = *a3;
  v8[1] = 0;
  v15 = 0;
  *v8 = v14;
  result = *(_QWORD **)(a1 + 40);
  v17 = *(_QWORD *)(a2 + 16);
  if ( result )
  {
LABEL_14:
    *(_QWORD *)(a1 + 40) = *result;
  }
  else
  {
LABEL_10:
    v38 = v4;
    v44 = v8;
    result = (_QWORD *)sub_145CBF0((__int64 *)(a1 + 48), 32, 8);
    v4 = v38;
    v8 = v44;
  }
  result[2] = *a3;
  v22 = *v4;
  *result = v17;
  result[3] = v22;
  result[1] = v15;
  v8[1] = (__int64)result;
  *(_QWORD *)(a2 + 16) = result;
  return result;
}
