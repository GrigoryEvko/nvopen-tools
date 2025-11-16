// Function: sub_10BFC20
// Address: 0x10bfc20
//
_QWORD *__fastcall sub_10BFC20(unsigned __int8 *a1, __int64 *a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  _QWORD *v4; // r11
  __int64 v6; // rax
  __int64 v7; // r14
  char v10; // al
  int v11; // r10d
  __int64 v12; // rax
  __int64 v13; // rbx
  unsigned int v14; // r10d
  __int64 v15; // r13
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // rcx
  __int64 v20; // rdx
  unsigned int v21; // r10d
  __int64 v22; // rax
  int v23; // esi
  __int64 v24; // r13
  __int64 *v25; // rax
  __int64 v26; // r13
  __int64 v27; // r12
  int v29; // esi
  __int64 v30; // rax
  __int64 v31; // [rsp+0h] [rbp-A0h]
  int v32; // [rsp+8h] [rbp-98h]
  int v33; // [rsp+8h] [rbp-98h]
  unsigned int v34; // [rsp+8h] [rbp-98h]
  __int64 v35; // [rsp+8h] [rbp-98h]
  unsigned int v36; // [rsp+8h] [rbp-98h]
  unsigned int v37; // [rsp+8h] [rbp-98h]
  unsigned __int8 *v38; // [rsp+10h] [rbp-90h] BYREF
  int v39; // [rsp+18h] [rbp-88h]
  __int64 v40[4]; // [rsp+20h] [rbp-80h] BYREF
  unsigned __int8 **v41; // [rsp+40h] [rbp-60h] BYREF
  char v42; // [rsp+48h] [rbp-58h]
  __int16 v43; // [rsp+60h] [rbp-40h]

  v2 = *((_QWORD *)a1 - 8);
  v3 = *(_QWORD *)(v2 + 16);
  if ( !v3 )
    return 0;
  v4 = *(_QWORD **)(v3 + 8);
  if ( v4 )
    return 0;
  if ( *(_BYTE *)v2 != 85 )
    return 0;
  v6 = *(_QWORD *)(v2 - 32);
  if ( !v6 || *(_BYTE *)v6 || *(_QWORD *)(v6 + 24) != *(_QWORD *)(v2 + 80) || (*(_BYTE *)(v6 + 33) & 0x20) == 0 )
    return 0;
  v7 = *((_QWORD *)a1 - 4);
  if ( *(_BYTE *)v7 == 85
    && (v19 = *(_QWORD *)(v7 - 32)) != 0
    && !*(_BYTE *)v19
    && *(_QWORD *)(v19 + 24) == *(_QWORD *)(v7 + 80)
    && (*(_BYTE *)(v19 + 33) & 0x20) != 0 )
  {
    v20 = *(_QWORD *)(v7 + 16);
    if ( !v20 )
      return 0;
    v4 = *(_QWORD **)(v20 + 8);
    if ( v4 )
      return 0;
    v21 = *(_DWORD *)(v6 + 36);
    if ( *(_DWORD *)(v19 + 36) != v21 )
      return 0;
    if ( v21 > 0xF )
    {
      if ( v21 - 180 > 1 )
        return v4;
      v36 = *(_DWORD *)(v6 + 36);
      if ( *(_QWORD *)(v7 + 32 * (2LL - (*(_DWORD *)(v7 + 4) & 0x7FFFFFF))) == *(_QWORD *)(v2
                                                                                         + 32
                                                                                         * (2LL
                                                                                          - (*(_DWORD *)(v2 + 4)
                                                                                           & 0x7FFFFFF))) )
      {
        v43 = 257;
        v22 = sub_10BBE20(
                a2,
                (unsigned int)*a1 - 29,
                *(_QWORD *)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF)),
                *(_QWORD *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF)),
                v40[0],
                0,
                (__int64)&v41,
                0);
        v23 = *a1;
        v31 = v22;
        v43 = 257;
        v24 = sub_10BBE20(
                a2,
                v23 - 29,
                *(_QWORD *)(v2 + 32 * (1LL - (*(_DWORD *)(v2 + 4) & 0x7FFFFFF))),
                *(_QWORD *)(v7 + 32 * (1LL - (*(_DWORD *)(v7 + 4) & 0x7FFFFFF))),
                v40[0],
                0,
                (__int64)&v41,
                0);
        v41 = (unsigned __int8 **)*((_QWORD *)a1 + 1);
        v25 = (__int64 *)sub_B43CA0((__int64)a1);
        v40[1] = v24;
        v26 = 0;
        v27 = sub_B6E160(v25, v36, (__int64)&v41, 1);
        v43 = 257;
        v40[0] = v31;
        v40[2] = *(_QWORD *)(v2 + 32 * (2LL - (*(_DWORD *)(v2 + 4) & 0x7FFFFFF)));
        if ( v27 )
          v26 = *(_QWORD *)(v27 + 24);
        v4 = sub_BD2CC0(88, 4u);
        if ( v4 )
        {
          v35 = (__int64)v4;
          sub_B44260((__int64)v4, **(_QWORD **)(v26 + 16), 56, 4u, 0, 0);
          *(_QWORD *)(v35 + 72) = 0;
          sub_B4A290(v35, v26, v27, v40, 3, (__int64)&v41, 0, 0);
          return (_QWORD *)v35;
        }
        return v4;
      }
      return 0;
    }
    if ( v21 <= 0xD )
      return v4;
    v29 = *a1;
    v43 = 257;
    v37 = v21;
    v30 = sub_10BBE20(
            a2,
            v29 - 29,
            *(_QWORD *)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF)),
            *(_QWORD *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF)),
            v39,
            0,
            (__int64)&v41,
            0);
    v14 = v37;
    v13 = v30;
  }
  else
  {
    v32 = *(_DWORD *)(v6 + 36);
    if ( (unsigned int)(v32 - 14) > 1 )
      return v4;
    v42 = 0;
    v41 = &v38;
    v10 = sub_991580((__int64)&v41, v7);
    v4 = 0;
    if ( !v10 )
      return v4;
    v43 = 257;
    if ( v32 == 15 )
    {
      sub_C496B0((__int64)v40, (__int64)v38);
      v11 = 15;
    }
    else
    {
      sub_C48440((__int64)v40, v38);
      v11 = v32;
    }
    v33 = v11;
    v12 = sub_AD8D80(*((_QWORD *)a1 + 1), (__int64)v40);
    v13 = sub_10BBE20(
            a2,
            (unsigned int)*a1 - 29,
            *(_QWORD *)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF)),
            v12,
            v39,
            0,
            (__int64)&v41,
            0);
    sub_969240(v40);
    v14 = v33;
  }
  v34 = v14;
  v15 = 0;
  v41 = (unsigned __int8 **)*((_QWORD *)a1 + 1);
  v16 = (__int64 *)sub_B43CA0((__int64)a1);
  v17 = sub_B6E160(v16, v34, (__int64)&v41, 1);
  v40[0] = v13;
  v43 = 257;
  v18 = v17;
  if ( v17 )
    v15 = *(_QWORD *)(v17 + 24);
  v4 = sub_BD2CC0(88, 2u);
  if ( v4 )
  {
    v35 = (__int64)v4;
    sub_B44260((__int64)v4, **(_QWORD **)(v15 + 16), 56, 2u, 0, 0);
    *(_QWORD *)(v35 + 72) = 0;
    sub_B4A290(v35, v15, v18, v40, 1, (__int64)&v41, 0, 0);
    return (_QWORD *)v35;
  }
  return v4;
}
