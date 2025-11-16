// Function: sub_DAF130
// Address: 0xdaf130
//
_QWORD *__fastcall sub_DAF130(__int64 *a1, unsigned __int64 *a2, __int64 a3, __int16 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 *v6; // r13
  _DWORD *v7; // rcx
  __int64 v8; // rdx
  unsigned __int64 v9; // rbx
  unsigned __int64 *v10; // r12
  int v11; // eax
  unsigned __int64 v12; // rbx
  __int64 v13; // rax
  _DWORD **v14; // rsi
  _QWORD *v15; // r12
  _DWORD *v16; // rdi
  __int64 v18; // rax
  unsigned __int64 *v19; // r12
  __int64 *v20; // rbx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 *v23; // r14
  __int64 v24; // rdx
  __int16 v25; // r13
  __int64 v26; // rax
  __int64 *v27; // r15
  __int64 *v28; // rax
  _QWORD *v29; // [rsp+8h] [rbp-158h]
  __int64 v30; // [rsp+18h] [rbp-148h]
  void *v31; // [rsp+20h] [rbp-140h]
  __int64 *v32; // [rsp+28h] [rbp-138h]
  int v34; // [rsp+38h] [rbp-128h]
  __int64 n; // [rsp+40h] [rbp-120h]
  __int64 *v38; // [rsp+68h] [rbp-F8h] BYREF
  __int64 v39; // [rsp+70h] [rbp-F0h] BYREF
  unsigned int v40; // [rsp+78h] [rbp-E8h]
  __int64 v41; // [rsp+80h] [rbp-E0h] BYREF
  unsigned int v42; // [rsp+88h] [rbp-D8h]
  __int64 v43; // [rsp+90h] [rbp-D0h] BYREF
  unsigned int v44; // [rsp+98h] [rbp-C8h]
  _DWORD *v45; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v46; // [rsp+A8h] [rbp-B8h]
  _DWORD v47[44]; // [rsp+B0h] [rbp-B0h] BYREF

  v46 = 0x2000000001LL;
  v6 = &a2[a3];
  v45 = v47;
  v47[0] = 5;
  n = 8 * a3;
  if ( a2 != v6 )
  {
    v7 = v47;
    v8 = 1;
    v9 = *a2;
    v10 = a2 + 1;
    v11 = *a2;
    while ( 1 )
    {
      v7[v8] = v11;
      v12 = HIDWORD(v9);
      LODWORD(v46) = v46 + 1;
      v13 = (unsigned int)v46;
      if ( (unsigned __int64)(unsigned int)v46 + 1 > HIDWORD(v46) )
      {
        sub_C8D5F0((__int64)&v45, v47, (unsigned int)v46 + 1LL, 4u, a5, a6);
        v13 = (unsigned int)v46;
      }
      v45[v13] = v12;
      v8 = (unsigned int)(v46 + 1);
      LODWORD(v46) = v46 + 1;
      if ( v6 == v10 )
        break;
      v9 = *v10;
      a5 = v8 + 1;
      v11 = *v10;
      if ( v8 + 1 > (unsigned __int64)HIDWORD(v46) )
      {
        v34 = *v10;
        sub_C8D5F0((__int64)&v45, v47, v8 + 1, 4u, a5, a6);
        v8 = (unsigned int)v46;
        v11 = v34;
      }
      v7 = v45;
      ++v10;
    }
  }
  v14 = &v45;
  v38 = 0;
  v15 = sub_C65B40((__int64)(a1 + 129), (__int64)&v45, (__int64 *)&v38, (__int64)off_49DEA80);
  if ( !v15 )
  {
    v18 = a1[133];
    a1[143] += n;
    v19 = (unsigned __int64 *)(a1 + 133);
    v20 = (__int64 *)((v18 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    if ( a1[134] >= (unsigned __int64)&v20[(unsigned __int64)n / 8] && v18 )
      a1[133] = (__int64)&v20[(unsigned __int64)n / 8];
    else
      v20 = (__int64 *)sub_9D1E70((__int64)v19, n, n, 3);
    if ( a2 != v6 )
      memmove(v20, a2, n);
    v32 = &v20[(unsigned __int64)n / 8];
    v31 = sub_C65D30((__int64)&v45, v19);
    v30 = v21;
    v22 = sub_A777F0(0x38u, (__int64 *)v19);
    v15 = (_QWORD *)v22;
    if ( !v22 )
      goto LABEL_40;
    v40 = 16;
    v39 = 1;
    if ( &v20[(unsigned __int64)n / 8] == v20 )
      goto LABEL_42;
    v29 = (_QWORD *)v22;
    v23 = v20;
    do
    {
      v24 = *v23;
      v44 = 16;
      v43 = *(unsigned __int16 *)(v24 + 26);
      sub_C49B30((__int64)&v41, (__int64)&v39, &v43);
      if ( v40 > 0x40 && v39 )
        j_j___libc_free_0_0(v39);
      v39 = v41;
      v40 = v42;
      if ( v44 > 0x40 && v43 )
        j_j___libc_free_0_0(v43);
      ++v23;
    }
    while ( v32 != v23 );
    v15 = v29;
    if ( v40 <= 0x40 )
    {
LABEL_42:
      v25 = v39;
    }
    else
    {
      v25 = *(_WORD *)v39;
      j_j___libc_free_0_0(v39);
    }
    *v15 = 0;
    *((_WORD *)v15 + 13) = v25;
    v15[1] = v31;
    *((_WORD *)v15 + 14) = 0;
    v15[2] = v30;
    *((_WORD *)v15 + 12) = 5;
    v15[4] = v20;
    v15[5] = a3;
    v26 = n >> 3;
    if ( n >> 5 > 0 )
    {
      v27 = &v20[4 * (n >> 5)];
      while ( *(_BYTE *)(sub_D95540(*v20) + 8) != 14 )
      {
        if ( *(_BYTE *)(sub_D95540(v20[1]) + 8) == 14 )
        {
          v32 = v20 + 1;
          goto LABEL_38;
        }
        if ( *(_BYTE *)(sub_D95540(v20[2]) + 8) == 14 )
        {
          v32 = v20 + 2;
          goto LABEL_38;
        }
        if ( *(_BYTE *)(sub_D95540(v20[3]) + 8) == 14 )
        {
          v32 = v20 + 3;
          goto LABEL_38;
        }
        v20 += 4;
        if ( v27 == v20 )
        {
          v26 = v32 - v20;
          goto LABEL_45;
        }
      }
      goto LABEL_37;
    }
LABEL_45:
    if ( v26 != 2 )
    {
      if ( v26 != 3 )
      {
        if ( v26 != 1 )
        {
LABEL_38:
          v28 = (__int64 *)v15[4];
          if ( v32 == &v28[v15[5]] )
            v15[6] = sub_D95540(*v28);
          else
            v15[6] = sub_D95540(*v32);
LABEL_40:
          sub_C657C0(a1 + 129, v15, v38, (__int64)off_49DEA80);
          v14 = (_DWORD **)v15;
          sub_DAEE00((__int64)a1, (__int64)v15, (__int64 *)a2, a3);
          goto LABEL_10;
        }
LABEL_48:
        if ( *(_BYTE *)(sub_D95540(*v20) + 8) != 14 )
          v20 = v32;
        v32 = v20;
        goto LABEL_38;
      }
      if ( *(_BYTE *)(sub_D95540(*v20) + 8) == 14 )
      {
LABEL_37:
        v32 = v20;
        goto LABEL_38;
      }
      ++v20;
    }
    if ( *(_BYTE *)(sub_D95540(*v20) + 8) != 14 )
    {
      ++v20;
      goto LABEL_48;
    }
    goto LABEL_37;
  }
LABEL_10:
  v16 = v45;
  *((_WORD *)v15 + 14) |= a4;
  if ( v16 != v47 )
    _libc_free(v16, v14);
  return v15;
}
