// Function: sub_1629D40
// Address: 0x1629d40
//
__int64 __fastcall sub_1629D40(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  int v3; // ecx
  __int64 v4; // r12
  __int64 v5; // rdx
  int v6; // edx
  unsigned int v7; // esi
  __int64 *v8; // r8
  __int64 v9; // rdi
  int v10; // eax
  __int64 result; // rax
  int v12; // esi
  int v13; // eax
  unsigned int v14; // esi
  __int64 *v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r9
  _QWORD *v18; // r9
  _QWORD *v19; // rcx
  __int64 v20; // r12
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 *v24; // r9
  __int64 v25; // rsi
  int v26; // r10d
  __int64 *v27; // r11
  _QWORD *v28; // r9
  _QWORD *v29; // rcx
  int v30; // eax
  __int64 *v31; // [rsp+8h] [rbp-B8h]
  unsigned int v32; // [rsp+14h] [rbp-ACh]
  int v33; // [rsp+14h] [rbp-ACh]
  _QWORD *v34; // [rsp+18h] [rbp-A8h]
  _QWORD *v35; // [rsp+20h] [rbp-A0h]
  __int64 v36; // [rsp+28h] [rbp-98h]
  int v37; // [rsp+30h] [rbp-90h]
  int v38; // [rsp+40h] [rbp-80h]
  __int64 v39; // [rsp+40h] [rbp-80h]
  __int64 v40; // [rsp+48h] [rbp-78h] BYREF
  int v41; // [rsp+5Ch] [rbp-64h] BYREF
  _QWORD *v42; // [rsp+60h] [rbp-60h] BYREF
  __int64 v43; // [rsp+68h] [rbp-58h]
  _QWORD *v44; // [rsp+70h] [rbp-50h]
  __int64 v45; // [rsp+78h] [rbp-48h]
  int v46; // [rsp+80h] [rbp-40h]
  int v47; // [rsp+84h] [rbp-3Ch] BYREF
  __int64 v48[7]; // [rsp+88h] [rbp-38h] BYREF

  v2 = a2;
  v40 = a1;
  v3 = *(_DWORD *)(a2 + 24);
  v42 = 0;
  v4 = *(_QWORD *)(a2 + 8);
  v43 = 0;
  v5 = *(unsigned int *)(a1 + 8);
  v38 = v3;
  v44 = (_QWORD *)(a1 + 8 * (1 - v5));
  v45 = (-8 * (1 - v5)) >> 3;
  v6 = *(_DWORD *)(a1 + 4);
  v46 = v6;
  v47 = *(unsigned __int16 *)(a1 + 2);
  v48[0] = *(_QWORD *)(a1 - 8LL * *(unsigned int *)(a1 + 8));
  if ( !v3 )
  {
LABEL_2:
    ++*(_QWORD *)v2;
    v7 = 0;
    goto LABEL_3;
  }
  v41 = v6;
  v12 = sub_15B64F0(&v41, &v47, v48);
  v13 = v38 - 1;
  v14 = (v38 - 1) & v12;
  v15 = (__int64 *)(v4 + 8LL * v14);
  v16 = *v15;
  if ( *v15 == -8 )
  {
LABEL_25:
    LODWORD(v20) = *(_DWORD *)(v2 + 24);
    v39 = *(_QWORD *)(v2 + 8);
    goto LABEL_26;
  }
  v37 = 1;
  v36 = v2;
  v34 = &v42[v43];
  v35 = &v44[v45];
  while ( 1 )
  {
    if ( v16 != -16 && v47 == *(unsigned __int16 *)(v16 + 2) )
    {
      v32 = *(_DWORD *)(v16 + 8);
      if ( v48[0] == *(_QWORD *)(v16 - 8LL * v32) && v46 == *(_DWORD *)(v16 + 4) )
      {
        v17 = v32 - 1;
        if ( !v43 )
        {
          if ( v45 != v17 )
            goto LABEL_13;
          v18 = (_QWORD *)(v16 + 8 * (1LL - v32));
          if ( v35 == v44 )
            goto LABEL_50;
          v33 = v13;
          v2 = v36;
          v19 = v44;
          v31 = v15;
          while ( *v19 == *v18 )
          {
            ++v19;
            ++v18;
            if ( v35 == v19 )
              goto LABEL_41;
          }
          goto LABEL_23;
        }
        if ( v43 == v17 )
          break;
      }
    }
LABEL_13:
    v14 = v13 & (v37 + v14);
    v15 = (__int64 *)(v4 + 8LL * v14);
    v16 = *v15;
    if ( *v15 == -8 )
    {
      v2 = v36;
      goto LABEL_25;
    }
    ++v37;
  }
  v28 = (_QWORD *)(v16 + 8 * (1LL - v32));
  if ( v34 != v42 )
  {
    v33 = v13;
    v2 = v36;
    v29 = v42;
    v31 = v15;
    while ( *v29 == *v28 )
    {
      ++v29;
      ++v28;
      if ( v34 == v29 )
      {
LABEL_41:
        v15 = v31;
        goto LABEL_42;
      }
    }
LABEL_23:
    v36 = v2;
    v13 = v33;
    goto LABEL_13;
  }
LABEL_50:
  v2 = v36;
LABEL_42:
  v39 = *(_QWORD *)(v2 + 8);
  v20 = *(unsigned int *)(v2 + 24);
  if ( (__int64 *)(v39 + 8 * v20) != v15 )
  {
    result = *v15;
    if ( *v15 )
      return result;
  }
LABEL_26:
  if ( !(_DWORD)v20 )
    goto LABEL_2;
  v42 = 0;
  v43 = 0;
  v21 = (-8 * (1LL - *(unsigned int *)(v40 + 8))) >> 3;
  v44 = (_QWORD *)(v40 + 8 * (1LL - *(unsigned int *)(v40 + 8)));
  v45 = v21;
  v46 = *(_DWORD *)(v40 + 4);
  v47 = *(unsigned __int16 *)(v40 + 2);
  v22 = *(_QWORD *)(v40 - 8LL * *(unsigned int *)(v40 + 8));
  v41 = v46;
  v48[0] = v22;
  v9 = v40;
  LODWORD(v23) = (v20 - 1) & sub_15B64F0(&v41, &v47, v48);
  v24 = (__int64 *)(v39 + 8LL * (unsigned int)v23);
  result = v40;
  v25 = *v24;
  if ( *v24 == v40 )
    return result;
  v26 = 1;
  v8 = 0;
  while ( v25 != -8 )
  {
    if ( v25 != -16 || v8 )
      v24 = v8;
    v23 = ((_DWORD)v20 - 1) & (unsigned int)(v23 + v26);
    v27 = (__int64 *)(v39 + 8 * v23);
    v25 = *v27;
    if ( *v27 == v40 )
      return result;
    ++v26;
    v8 = v24;
    v24 = (__int64 *)(v39 + 8 * v23);
  }
  v30 = *(_DWORD *)(v2 + 16);
  v7 = *(_DWORD *)(v2 + 24);
  if ( !v8 )
    v8 = v24;
  ++*(_QWORD *)v2;
  v10 = v30 + 1;
  if ( 4 * v10 >= 3 * v7 )
  {
LABEL_3:
    v7 *= 2;
    goto LABEL_4;
  }
  if ( v7 - (v10 + *(_DWORD *)(v2 + 20)) <= v7 >> 3 )
  {
LABEL_4:
    sub_15BA380(v2, v7);
    sub_15B7230(v2, &v40, &v42);
    v8 = v42;
    v9 = v40;
    v10 = *(_DWORD *)(v2 + 16) + 1;
  }
  *(_DWORD *)(v2 + 16) = v10;
  if ( *v8 != -8 )
    --*(_DWORD *)(v2 + 20);
  *v8 = v9;
  return v40;
}
