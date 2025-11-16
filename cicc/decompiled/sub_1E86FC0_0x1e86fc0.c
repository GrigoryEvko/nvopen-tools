// Function: sub_1E86FC0
// Address: 0x1e86fc0
//
char __fastcall sub_1E86FC0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, signed int a6, int a7)
{
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r9
  unsigned int *v15; // r9
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdi
  char v19; // r9
  unsigned __int8 v20; // dl
  __int64 v21; // rdx
  __int64 v22; // r11
  unsigned int v23; // esi
  _WORD *v24; // r10
  __int16 v25; // dx
  _WORD *v26; // rsi
  unsigned __int16 v27; // dx
  __int16 v28; // r10
  unsigned int *v30; // [rsp+0h] [rbp-60h]
  char v31; // [rsp+0h] [rbp-60h]
  unsigned int *v32; // [rsp+0h] [rbp-60h]
  char v34[16]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v35; // [rsp+20h] [rbp-40h]

  v12 = (__int64 *)sub_1DB3C70((__int64 *)a5, a4);
  if ( v12 == (__int64 *)(*(_QWORD *)a5 + 24LL * *(unsigned int *)(a5 + 8))
    || (*(_DWORD *)((*v12 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v12 >> 1) & 3) > (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                          | (unsigned int)(a4 >> 1) & 3)
    || (v14 = v12[2]) == 0 )
  {
    sub_1E86D40(a1, "No live segment at def", a2, a3, 0);
    sub_1E85C60(a5);
    if ( a6 < 0 )
      sub_1E85940(a1, a6);
    else
      sub_1E859F0(a1, a6);
    if ( a7 )
      sub_1E85CD0(a7);
    sub_1E85AA0(a4);
  }
  else
  {
    v30 = (unsigned int *)v12[2];
    if ( *(_QWORD *)(v14 + 8) != a4 )
    {
      sub_1E86D40(a1, "Inconsistent valno->def", a2, a3, 0);
      sub_1E85C60(a5);
      if ( a6 < 0 )
      {
        sub_1E85940(a1, a6);
        v15 = v30;
        if ( !a7 )
          goto LABEL_14;
      }
      else
      {
        sub_1E859F0(a1, a6);
        v15 = v30;
        if ( !a7 )
        {
LABEL_14:
          sub_1E85BF0(v15);
          sub_1E85AA0(a4);
          goto LABEL_8;
        }
      }
      v32 = v15;
      sub_1E85CD0(a7);
      v15 = v32;
      goto LABEL_14;
    }
  }
LABEL_8:
  LOBYTE(v13) = *(_BYTE *)(a2 + 3) >> 6;
  if ( ((*(_BYTE *)(a2 + 3) >> 4) & ((*(_BYTE *)(a2 + 3) & 0x40) != 0)) == 0 )
    return v13;
  v31 = (*(_BYTE *)(a2 + 3) >> 4) & ((*(_BYTE *)(a2 + 3) & 0x40) != 0);
  sub_1E86030((__int64)v34, a5, a4);
  LOBYTE(v13) = v35 ^ 6;
  if ( (((unsigned __int8)v35 ^ 6) & 6) == 0 )
    return v13;
  if ( a6 < 0 )
  {
    sub_1E86D40(a1, "Live range continues after dead def flag", a2, a3, 0);
    sub_1E85C60(a5);
    sub_1E85940(a1, a6);
LABEL_33:
    LOBYTE(v13) = a7;
    if ( a7 )
      LOBYTE(v13) = sub_1E85CD0(a7);
    return v13;
  }
  v16 = *(_QWORD *)(a2 + 16);
  v17 = *(_QWORD *)(v16 + 32);
  v13 = 5LL * *(unsigned int *)(v16 + 40);
  v18 = v17 + 8 * v13;
  if ( v17 == v18 )
    goto LABEL_31;
  v19 = 0;
  do
  {
LABEL_20:
    if ( !*(_BYTE *)v17 )
    {
      LOBYTE(v13) = *(_BYTE *)(v17 + 3);
      if ( (v13 & 0x10) != 0 )
      {
        v20 = (unsigned __int8)v13 >> 6;
        LOBYTE(v13) = (v13 & 0x10) != 0;
        if ( ((unsigned __int8)v13 & v20) == 0 )
        {
          v21 = *(_QWORD *)(a1 + 40);
          if ( !v21 )
            BUG();
          v22 = *(unsigned int *)(v17 + 8);
          v23 = *(_DWORD *)(*(_QWORD *)(v21 + 8) + 24 * v22 + 16);
          v24 = (_WORD *)(*(_QWORD *)(v21 + 56) + 2LL * (v23 >> 4));
          v25 = v22 * (v23 & 0xF);
          v26 = v24 + 1;
          v27 = *v24 + v25;
LABEL_26:
          v13 = (__int64)v26;
          while ( v13 )
          {
            if ( a6 == v27 )
            {
              v17 += 40;
              v19 = v31;
              if ( v18 != v17 )
                goto LABEL_20;
              goto LABEL_30;
            }
            v28 = *(_WORD *)v13;
            v26 = 0;
            v13 += 2;
            v27 += v28;
            if ( !v28 )
              goto LABEL_26;
          }
        }
      }
    }
    v17 += 40;
  }
  while ( v18 != v17 );
LABEL_30:
  if ( !v19 )
  {
LABEL_31:
    sub_1E86D40(a1, "Live range continues after dead def flag", a2, a3, 0);
    sub_1E85C60(a5);
    sub_1E859F0(a1, a6);
    goto LABEL_33;
  }
  return v13;
}
