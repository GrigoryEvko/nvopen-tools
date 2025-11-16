// Function: sub_DCC290
// Address: 0xdcc290
//
__int64 __fastcall sub_DCC290(__int64 *a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // r13
  __int64 v5; // r12
  __int64 *v7; // rdi
  __int64 v8; // rbx
  unsigned int v9; // r15d
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 *v12; // rax
  unsigned __int64 v13; // rsi
  __int64 v14; // r8
  unsigned __int64 i; // rax
  int v16; // ebx
  char *v17; // rdx
  char *v18; // rdx
  __int64 *v19; // r12
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // rdi
  const void *v23; // r12
  const void *v24; // rdx
  const void *v25; // rbx
  __int64 v26; // r9
  int v27; // eax
  __int64 v28; // r8
  __int64 *v29; // rsi
  __int64 v30; // [rsp+0h] [rbp-70h]
  _QWORD *v31; // [rsp+8h] [rbp-68h]
  int v32; // [rsp+8h] [rbp-68h]
  __int64 v33; // [rsp+8h] [rbp-68h]
  unsigned __int64 v34; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v35; // [rsp+18h] [rbp-58h]
  _QWORD *p_dest; // [rsp+20h] [rbp-50h] BYREF
  __int64 v37; // [rsp+28h] [rbp-48h]
  _QWORD *dest; // [rsp+30h] [rbp-40h] BYREF
  char v39[56]; // [rsp+38h] [rbp-38h] BYREF

  v4 = (_QWORD *)a3;
  v5 = a2;
  if ( *(_WORD *)(a2 + 24) != 6 || (*(_BYTE *)(a2 + 28) & 2) == 0 )
    return sub_DCB270((__int64)a1, v5, (__int64)v4);
  if ( *(_WORD *)(a3 + 24) )
    goto LABEL_21;
  v7 = *(__int64 **)(a2 + 32);
  v8 = *v7;
  if ( *(_WORD *)(*v7 + 24) )
    goto LABEL_21;
  if ( a3 != v8 )
  {
    sub_D96BA0((__int64)&v34, *v7, a3);
    v9 = v35;
    if ( v35 > 0x40 )
    {
      if ( v9 - (unsigned int)sub_C444A0((__int64)&v34) <= 1 )
      {
LABEL_19:
        if ( v34 )
          j_j___libc_free_0_0(v34);
        goto LABEL_21;
      }
LABEL_9:
      sub_C4A1D0((__int64)&p_dest, *(_QWORD *)(v8 + 32) + 24LL, (__int64)&v34);
      v31 = sub_DA26C0(a1, (__int64)&p_dest);
      if ( (unsigned int)v37 > 0x40 && p_dest )
        j_j___libc_free_0_0(p_dest);
      sub_C4A1D0((__int64)&p_dest, v4[4] + 24LL, (__int64)&v34);
      v4 = sub_DA26C0(a1, (__int64)&p_dest);
      if ( (unsigned int)v37 > 0x40 && p_dest )
        j_j___libc_free_0_0(p_dest);
      v10 = *(_QWORD *)(a2 + 40);
      dest = v31;
      v37 = 0x200000001LL;
      v11 = *(_QWORD *)(a2 + 32);
      p_dest = &dest;
      sub_D932D0((__int64)&p_dest, v39, (char *)(v11 + 8), (char *)(v11 + 8 * v10));
      v12 = sub_DC8BD0(a1, (__int64)&p_dest, 0, 0);
      v5 = (__int64)v12;
      if ( *((_WORD *)v12 + 12) != 6 )
      {
        v29 = v12;
        v19 = (__int64 *)sub_DCC290(a1, v12, v4);
        if ( p_dest != &dest )
          _libc_free(p_dest, v29);
        if ( v35 > 0x40 && v34 )
          j_j___libc_free_0_0(v34);
        return (__int64)v19;
      }
      if ( p_dest != &dest )
        _libc_free(p_dest, &p_dest);
      if ( v35 <= 0x40 )
        goto LABEL_21;
      goto LABEL_19;
    }
    if ( v34 )
    {
      _BitScanReverse64(&v20, v34);
      if ( ((unsigned int)v20 ^ 0x3F) != 0x3F )
        goto LABEL_9;
    }
LABEL_21:
    v13 = *(_QWORD *)(v5 + 40);
    if ( !(_DWORD)v13 )
      return sub_DCB270((__int64)a1, v5, (__int64)v4);
    v14 = *(_QWORD *)(v5 + 32);
    for ( i = 0; ; ++i )
    {
      v16 = i + 1;
      if ( *(_QWORD **)(v14 + 8 * i) == v4 )
        break;
      if ( i == (_DWORD)v13 - 1 )
        return sub_DCB270((__int64)a1, v5, (__int64)v4);
    }
    v17 = *(char **)(v5 + 32);
    if ( v13 <= i )
      i = *(_QWORD *)(v5 + 40);
    p_dest = &dest;
    v37 = 0x200000000LL;
    sub_D932D0((__int64)&p_dest, (char *)&dest, v17, (char *)(v14 + 8 * i));
    v18 = (char *)(*(_QWORD *)(v5 + 32) + 8LL * v16);
    sub_D932D0((__int64)&p_dest, (char *)&p_dest[(unsigned int)v37], v18, &v18[8 * (*(_QWORD *)(v5 + 40) - v16)]);
    goto LABEL_29;
  }
  v21 = sub_D91800((__int64)v7, *(_QWORD *)(a2 + 40), 1);
  p_dest = &dest;
  v22 = &dest;
  v23 = (const void *)v21;
  v25 = v24;
  v26 = (__int64)v24 - v21;
  v37 = 0x200000000LL;
  v27 = 0;
  v28 = v26 >> 3;
  if ( (unsigned __int64)v26 > 0x10 )
  {
    v30 = v26;
    v33 = v26 >> 3;
    sub_C8D5F0((__int64)&p_dest, &dest, v26 >> 3, 8u, v28, v26);
    v27 = v37;
    v26 = v30;
    LODWORD(v28) = v33;
    v22 = &p_dest[(unsigned int)v37];
  }
  if ( v25 != v23 )
  {
    v32 = v28;
    memcpy(v22, v23, v26);
    v27 = v37;
    LODWORD(v28) = v32;
  }
  LODWORD(v37) = v28 + v27;
LABEL_29:
  v19 = sub_DC8BD0(a1, (__int64)&p_dest, 0, 0);
  if ( p_dest != &dest )
    _libc_free(p_dest, &p_dest);
  return (__int64)v19;
}
