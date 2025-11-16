// Function: sub_9B7FB0
// Address: 0x9b7fb0
//
__int64 __fastcall sub_9B7FB0(unsigned int a1, _DWORD *a2, __int64 a3, __int64 *a4, __int64 *a5, __int64 *a6, char a7)
{
  unsigned int v11; // eax
  __int64 v12; // rdx
  unsigned int v13; // esi
  int v14; // eax
  _DWORD *v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rdi
  _DWORD *v18; // rax
  _DWORD *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rdx
  int v24; // ecx
  unsigned int v25; // r10d
  unsigned __int64 v27; // rax
  __int64 *v28; // [rsp+8h] [rbp-58h]
  __int64 *v29; // [rsp+10h] [rbp-50h]
  __int64 *v30; // [rsp+10h] [rbp-50h]
  __int64 *v31; // [rsp+10h] [rbp-50h]
  __int64 *v32; // [rsp+10h] [rbp-50h]
  __int64 *v33; // [rsp+18h] [rbp-48h]
  __int64 *v34; // [rsp+18h] [rbp-48h]
  __int64 *v35; // [rsp+18h] [rbp-48h]
  __int64 v36; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v37; // [rsp+28h] [rbp-38h]

  v37 = a1;
  if ( a1 > 0x40 )
  {
    v32 = a6;
    v35 = a5;
    sub_C43690(&v36, 0, 0);
    a6 = v32;
    a5 = v35;
  }
  else
  {
    v36 = 0;
  }
  if ( *((_DWORD *)a6 + 2) > 0x40u && *a6 )
  {
    v29 = a6;
    v33 = a5;
    j_j___libc_free_0_0(*a6);
    a6 = v29;
    a5 = v33;
  }
  v11 = v37;
  v12 = v36;
  v37 = 0;
  *a6 = v36;
  *((_DWORD *)a6 + 2) = v11;
  if ( *((_DWORD *)a5 + 2) <= 0x40u && v11 <= 0x40 )
  {
    *a5 = v12;
    *((_DWORD *)a5 + 2) = *((_DWORD *)a6 + 2);
    v13 = *((_DWORD *)a4 + 2);
    if ( v13 > 0x40 )
      goto LABEL_12;
  }
  else
  {
    v30 = a6;
    v34 = a5;
    sub_C43990(a5, a6);
    a5 = v34;
    a6 = v30;
    if ( v37 > 0x40 && v36 )
    {
      j_j___libc_free_0_0(v36);
      a6 = v30;
      a5 = v34;
    }
    v13 = *((_DWORD *)a4 + 2);
    if ( v13 > 0x40 )
    {
LABEL_12:
      v28 = a6;
      v31 = a5;
      v14 = sub_C444A0(a4);
      a5 = v31;
      a6 = v28;
      if ( v13 != v14 )
        goto LABEL_13;
      return 1;
    }
  }
  if ( !*a4 )
    return 1;
LABEL_13:
  v15 = &a2[a3];
  v16 = (4 * a3) >> 4;
  v17 = (4 * a3) >> 2;
  if ( v16 > 0 )
  {
    v18 = a2;
    v19 = &a2[4 * v16];
    while ( 1 )
    {
      if ( *v18 )
        goto LABEL_20;
      if ( v18[1] )
      {
        if ( v15 != v18 + 1 )
          goto LABEL_21;
        goto LABEL_44;
      }
      if ( v18[2] )
      {
        if ( v15 != v18 + 2 )
          goto LABEL_21;
        goto LABEL_44;
      }
      if ( v18[3] )
        break;
      v18 += 4;
      if ( v18 == v19 )
      {
        v17 = v15 - v18;
        goto LABEL_41;
      }
    }
    if ( v15 == v18 + 3 )
      goto LABEL_44;
LABEL_21:
    if ( (_DWORD)a3 )
    {
      v20 = 0;
      while ( 1 )
      {
        v23 = *a4;
        if ( v13 > 0x40 )
          v23 = *(_QWORD *)(v23 + 8LL * ((unsigned int)v20 >> 6));
        if ( (v23 & (1LL << v20)) == 0 )
          goto LABEL_25;
        v24 = a2[v20];
        v25 = (unsigned int)v24 >> 31;
        LOBYTE(v25) = a7 & (v24 < 0);
        if ( (_BYTE)v25 )
          goto LABEL_25;
        if ( v24 < 0 )
          return v25;
        if ( (int)a1 > v24 )
          break;
        v24 -= a1;
        v21 = *a6;
        v22 = 1LL << v24;
        if ( *((_DWORD *)a6 + 2) > 0x40u )
          goto LABEL_46;
        ++v20;
        *a6 = v22 | v21;
        if ( (unsigned int)a3 == v20 )
          return 1;
LABEL_26:
        v13 = *((_DWORD *)a4 + 2);
      }
      v21 = *a5;
      v22 = 1LL << v24;
      if ( *((_DWORD *)a5 + 2) > 0x40u )
LABEL_46:
        *(_QWORD *)(v21 + 8LL * ((unsigned int)v24 >> 6)) |= v22;
      else
        *a5 = v22 | v21;
LABEL_25:
      if ( (unsigned int)a3 == ++v20 )
        return 1;
      goto LABEL_26;
    }
    return 1;
  }
  v18 = a2;
LABEL_41:
  if ( v17 != 2 )
  {
    if ( v17 != 3 )
    {
      if ( v17 != 1 )
        goto LABEL_44;
      goto LABEL_59;
    }
    if ( *v18 )
      goto LABEL_20;
    ++v18;
  }
  if ( *v18 )
    goto LABEL_20;
  ++v18;
LABEL_59:
  if ( *v18 )
  {
LABEL_20:
    if ( v15 != v18 )
      goto LABEL_21;
  }
LABEL_44:
  v27 = *a5;
  if ( *((_DWORD *)a5 + 2) > 0x40u )
  {
    *(_QWORD *)v27 |= 1uLL;
    return 1;
  }
  else
  {
    v25 = 1;
    *a5 = v27 | 1;
  }
  return v25;
}
