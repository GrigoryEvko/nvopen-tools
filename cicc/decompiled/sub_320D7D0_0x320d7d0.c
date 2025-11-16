// Function: sub_320D7D0
// Address: 0x320d7d0
//
__int64 __fastcall sub_320D7D0(__int64 a1)
{
  __int64 *v2; // rbx
  unsigned int v3; // r12d
  __int64 v4; // r8
  __int64 v5; // r10
  int v6; // esi
  unsigned int i; // eax
  _QWORD *v8; // rdx
  unsigned int v9; // eax
  unsigned __int8 v10; // al
  __int64 v11; // rdx
  _QWORD *v12; // rdx
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // r9
  _QWORD *v15; // r11
  _QWORD *v16; // rax
  unsigned __int64 v17; // rcx
  __int64 v18; // r14
  char *v19; // r14
  unsigned __int64 v20; // r13
  unsigned __int64 v21; // rdi
  __int64 *v23; // rdx
  __int64 v24; // r9
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdx
  _QWORD *v27; // r10
  _QWORD *v28; // rax
  _QWORD *v29; // rsi
  __int64 *v30; // [rsp+0h] [rbp-C0h]
  __int64 v31; // [rsp+8h] [rbp-B8h]
  __int64 v32; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v33; // [rsp+18h] [rbp-A8h]
  __int64 v34; // [rsp+20h] [rbp-A0h]
  __int64 v35; // [rsp+28h] [rbp-98h]
  __int64 v36[2]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v37; // [rsp+40h] [rbp-80h]
  __int64 v38; // [rsp+48h] [rbp-78h]
  unsigned int v39; // [rsp+50h] [rbp-70h]
  char *v40; // [rsp+58h] [rbp-68h]
  __int64 v41; // [rsp+60h] [rbp-60h]
  char v42; // [rsp+68h] [rbp-58h] BYREF
  unsigned __int64 v43; // [rsp+70h] [rbp-50h]
  unsigned int v44; // [rsp+78h] [rbp-48h]
  char v45; // [rsp+80h] [rbp-40h]

  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  sub_320D160(a1, (__int64)&v32);
  v2 = *(__int64 **)(a1 + 368);
  v30 = &v2[12 * *(unsigned int *)(a1 + 376)];
  if ( v2 != v30 )
  {
    v3 = v35;
    v31 = v33;
    while ( 1 )
    {
      v4 = *v2;
      v5 = v2[1];
      if ( v3 )
      {
        v6 = 1;
        for ( i = (v3 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)
                    | ((unsigned __int64)(((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)) << 32))) >> 31)
                 ^ (484763065 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)))); ; i = (v3 - 1) & v9 )
        {
          v8 = (_QWORD *)(v31 + 16LL * i);
          if ( *v8 == v4 && v8[1] == v5 )
            break;
          if ( *v8 == -4096 && v8[1] == -4096 )
            goto LABEL_9;
          v9 = v6 + i;
          ++v6;
        }
        goto LABEL_30;
      }
LABEL_9:
      v10 = *(_BYTE *)(v4 - 16);
      v11 = v4 - 16;
      if ( v5 )
        break;
      if ( (*(_BYTE *)(v4 - 16) & 2) != 0 )
        v23 = *(__int64 **)(v4 - 32);
      else
        v23 = (__int64 *)(v11 - 8LL * ((v10 >> 2) & 0xF));
      v24 = *v23;
      v25 = *(_QWORD *)(a1 + 88);
      v26 = *v23 % v25;
      v27 = *(_QWORD **)(*(_QWORD *)(a1 + 80) + 8 * v26);
      if ( v27 )
      {
        v28 = (_QWORD *)*v27;
        if ( *(_QWORD *)(*v27 + 8LL) != v24 )
        {
          do
          {
            v29 = (_QWORD *)*v28;
            if ( !*v28 )
              goto LABEL_30;
            v27 = v28;
            if ( v26 != v29[1] % v25 )
              goto LABEL_30;
            v28 = (_QWORD *)*v28;
          }
          while ( v29[1] != v24 );
        }
        v18 = *v27 + 16LL;
        if ( *v27 )
          goto LABEL_21;
      }
LABEL_30:
      v2 += 12;
      if ( v30 == v2 )
        return sub_C7D6A0(v31, 16LL * v3, 8);
    }
    if ( (*(_BYTE *)(v4 - 16) & 2) != 0 )
      v12 = *(_QWORD **)(v4 - 32);
    else
      v12 = (_QWORD *)(v11 - 8LL * ((v10 >> 2) & 0xF));
    v13 = *(_QWORD *)(a1 + 144);
    v14 = v5 + 31LL * *v12;
    v15 = *(_QWORD **)(*(_QWORD *)(a1 + 136) + 8 * (v14 % v13));
    if ( !v15 )
      goto LABEL_30;
    v16 = (_QWORD *)*v15;
    v17 = *(_QWORD *)(*v15 + 208LL);
    while ( v14 != v17 || *v12 != v16[1] || v16[2] != v5 )
    {
      if ( !*v16 )
        goto LABEL_30;
      v17 = *(_QWORD *)(*v16 + 208LL);
      v15 = v16;
      if ( v14 % v13 != v17 % v13 )
        goto LABEL_30;
      v16 = (_QWORD *)*v16;
    }
    v18 = *v15 + 24LL;
    if ( !*v15 )
      goto LABEL_30;
LABEL_21:
    v42 = 0;
    v36[0] = v4;
    v37 = 0;
    v36[1] = 0;
    v38 = 0;
    v39 = 0;
    v40 = &v42;
    v41 = 0;
    v45 = 0;
    sub_3202DE0(a1, (__int64)v36, (__int64)(v2 + 2));
    sub_320CF50(a1, v36, v18);
    if ( v45 )
    {
      v45 = 0;
      if ( v44 > 0x40 )
      {
        if ( v43 )
          j_j___libc_free_0_0(v43);
      }
    }
    v19 = v40;
    v20 = (unsigned __int64)&v40[40 * (unsigned int)v41];
    if ( v40 != (char *)v20 )
    {
      do
      {
        v20 -= 40LL;
        v21 = *(_QWORD *)(v20 + 8);
        if ( v21 != v20 + 24 )
          _libc_free(v21);
      }
      while ( v19 != (char *)v20 );
      v20 = (unsigned __int64)v40;
    }
    if ( (char *)v20 != &v42 )
      _libc_free(v20);
    sub_C7D6A0(v37, 12LL * v39, 4);
    v3 = v35;
    v31 = v33;
    goto LABEL_30;
  }
  return sub_C7D6A0(v33, 16LL * (unsigned int)v35, 8);
}
