// Function: sub_11ED480
// Address: 0x11ed480
//
__int64 __fastcall sub_11ED480(__int64 **a1, unsigned __int8 *a2, __int64 a3)
{
  int v5; // edx
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // rdx
  int v11; // r15d
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  void *v15; // r10
  __int64 v16; // r8
  _QWORD *v17; // rdx
  _QWORD *v18; // r15
  _QWORD *v19; // rbx
  __int64 v20; // rax
  __int64 v21; // r9
  _QWORD *v22; // rax
  int v23; // edx
  char *v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // r13
  __int64 *v29; // [rsp-10h] [rbp-C0h]
  __int64 v30; // [rsp+8h] [rbp-A8h]
  __int64 v31; // [rsp+10h] [rbp-A0h]
  void *v32; // [rsp+18h] [rbp-98h]
  void *v33; // [rsp+18h] [rbp-98h]
  __int64 v34; // [rsp+28h] [rbp-88h]
  __int64 v35; // [rsp+30h] [rbp-80h] BYREF
  __int64 v36; // [rsp+38h] [rbp-78h]

  BYTE4(v34) = 0;
  v35 = 0x100000002LL;
  if ( !sub_11EC990((__int64)a1, (__int64)a2, 3u, 0x100000001LL, v34, 0x100000002LL) )
    return 0;
  v5 = *a2;
  if ( v5 != 40 )
  {
    v6 = -32;
    if ( v5 != 85 )
    {
      v6 = -96;
      if ( v5 != 34 )
        BUG();
    }
    if ( (a2[7] & 0x80u) == 0 )
      goto LABEL_11;
    goto LABEL_5;
  }
  v6 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  if ( (a2[7] & 0x80u) != 0 )
  {
LABEL_5:
    v7 = sub_BD2BC0((__int64)a2);
    v9 = v7 + v8;
    v10 = 0;
    if ( (a2[7] & 0x80u) != 0 )
      v10 = sub_BD2BC0((__int64)a2);
    if ( (unsigned int)((v9 - v10) >> 4) )
    {
      if ( (a2[7] & 0x80u) == 0 )
        BUG();
      v11 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
      if ( (a2[7] & 0x80u) == 0 )
        BUG();
      v12 = sub_BD2BC0((__int64)a2);
      v6 -= 32LL * (unsigned int)(*(_DWORD *)(v12 + v13 - 4) - v11);
    }
  }
LABEL_11:
  v14 = sub_11D9DD0((__int64)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)], (__int64)&a2[v6], 5);
  v35 = (__int64)v15;
  v16 = v14;
  v18 = v17;
  v36 = 0x800000000LL;
  v19 = (_QWORD *)v14;
  v20 = (__int64)v17 - v14;
  v21 = v20 >> 5;
  if ( (unsigned __int64)v20 > 0x100 )
  {
    v30 = v16;
    v31 = v20 >> 5;
    v33 = v15;
    sub_C8D5F0((__int64)&v35, v15, v20 >> 5, 8u, v16, v21);
    v24 = (char *)v35;
    v23 = v36;
    v15 = v33;
    LODWORD(v21) = v31;
    v16 = v30;
    v22 = (_QWORD *)(v35 + 8LL * (unsigned int)v36);
  }
  else
  {
    v22 = v15;
    v23 = 0;
    v24 = (char *)v15;
  }
  if ( v18 != (_QWORD *)v16 )
  {
    do
    {
      if ( v22 )
        *v22 = *v19;
      v19 += 4;
      ++v22;
    }
    while ( v18 != v19 );
    v24 = (char *)v35;
    v23 = v36;
  }
  v29 = *a1;
  v25 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
  LODWORD(v36) = v21 + v23;
  v32 = v15;
  v26 = *(_QWORD *)&a2[32 * (1 - v25)];
  v27 = sub_11CAB60(
          *(_QWORD *)&a2[-32 * v25],
          v26,
          *(_QWORD *)&a2[32 * (4 - v25)],
          v24,
          (unsigned int)(v21 + v23),
          a3,
          v29);
  if ( v27 && *(_BYTE *)v27 == 85 )
    *(_WORD *)(v27 + 2) = *(_WORD *)(v27 + 2) & 0xFFFC | *((_WORD *)a2 + 1) & 3;
  if ( (void *)v35 != v32 )
    _libc_free(v35, v26);
  return v27;
}
