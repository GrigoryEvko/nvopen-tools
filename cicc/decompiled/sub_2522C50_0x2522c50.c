// Function: sub_2522C50
// Address: 0x2522c50
//
__int64 __fastcall sub_2522C50(__int64 a1, unsigned __int64 *a2, __int64 a3, _QWORD *a4, _BYTE *a5, char a6, int a7)
{
  __int64 result; // rax
  unsigned __int8 *v11; // r10
  int v12; // eax
  unsigned __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  int v18; // edx
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // rax
  unsigned int v21; // eax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  unsigned __int8 *v27; // rax
  __int64 v28; // rbx
  __int64 v29; // rax
  __int64 v30; // rbx
  __int64 (*v31)(); // rax
  __int64 v32; // [rsp-78h] [rbp-78h]
  __int64 v33; // [rsp-68h] [rbp-68h]
  unsigned __int8 *v34; // [rsp-60h] [rbp-60h]
  __int64 v35; // [rsp-60h] [rbp-60h]
  int v36; // [rsp-60h] [rbp-60h]
  unsigned __int8 *v37; // [rsp-60h] [rbp-60h]
  __int64 v38; // [rsp-58h] [rbp-58h]
  unsigned __int8 v40; // [rsp-4Ch] [rbp-4Ch]
  unsigned __int64 v41; // [rsp-48h] [rbp-48h] BYREF
  __int64 v42; // [rsp-40h] [rbp-40h]

  result = *(unsigned __int8 *)(a1 + 4300);
  if ( !(_BYTE)result )
    return result;
  v11 = (unsigned __int8 *)a2[3];
  v12 = *v11;
  if ( (unsigned __int8)v12 <= 0x1Cu )
  {
    v41 = sub_250D2C0(*a2, 0);
    v42 = v15;
    return sub_251C230(a1, (__int64 *)&v41, a3, a4, a5, a6, a7);
  }
  if ( (unsigned __int8)(v12 - 34) <= 0x33u )
  {
    v14 = 0x8000000000041LL;
    if ( !_bittest64(&v14, (unsigned int)(v12 - 34)) )
      goto LABEL_13;
    if ( a2 < (unsigned __int64 *)&v11[-32 * (*((_DWORD *)v11 + 1) & 0x7FFFFFF)] )
      goto LABEL_10;
    if ( v12 == 40 )
    {
      v34 = (unsigned __int8 *)a2[3];
      v21 = sub_B491D0((__int64)v11);
      v11 = v34;
      v38 = -32 - 32LL * v21;
    }
    else
    {
      v38 = -32;
      if ( v12 != 85 )
      {
        v38 = -96;
        if ( v12 != 34 )
          BUG();
      }
    }
    if ( (v11[7] & 0x80u) != 0 )
    {
      v33 = (__int64)v11;
      v22 = sub_BD2BC0((__int64)v11);
      v11 = (unsigned __int8 *)v33;
      v35 = v23 + v22;
      if ( *(char *)(v33 + 7) >= 0 )
      {
        if ( !(unsigned int)(v35 >> 4) )
          goto LABEL_30;
      }
      else
      {
        v24 = sub_BD2BC0(v33);
        v11 = (unsigned __int8 *)v33;
        if ( !(unsigned int)((v35 - v24) >> 4) )
          goto LABEL_30;
        if ( *(char *)(v33 + 7) < 0 )
        {
          v36 = *(_DWORD *)(sub_BD2BC0(v33) + 8);
          if ( *(char *)(v33 + 7) >= 0 )
            BUG();
          v25 = sub_BD2BC0(v33);
          v11 = (unsigned __int8 *)v33;
          v38 -= 32LL * (unsigned int)(*(_DWORD *)(v25 + v26 - 4) - v36);
          goto LABEL_30;
        }
      }
      BUG();
    }
LABEL_30:
    if ( a2 < (unsigned __int64 *)&v11[v38] )
    {
      v27 = &v11[-32 * (*((_DWORD *)v11 + 1) & 0x7FFFFFF)];
      v28 = ((char *)a2 - (char *)v27) >> 5;
      if ( (v11[7] & 0x40) != 0 )
        v27 = (unsigned __int8 *)*((_QWORD *)v11 - 1);
      v42 = 0;
      v41 = (unsigned __int64)&v27[32 * (unsigned int)v28] | 3;
      nullsub_1518();
      return sub_251C230(a1, (__int64 *)&v41, a3, a4, a5, a6, a7);
    }
    goto LABEL_10;
  }
  if ( (_BYTE)v12 == 30 )
  {
    v13 = sub_B43CB0((__int64)v11);
    sub_250D230(&v41, v13, 2, 0);
    return sub_251C230(a1, (__int64 *)&v41, a3, a4, a5, a6, a7);
  }
LABEL_13:
  if ( (_BYTE)v12 == 84 )
  {
    v16 = *(_QWORD *)(*((_QWORD *)v11 - 1)
                    + 32LL * *((unsigned int *)v11 + 18)
                    + 8LL * (unsigned int)(((__int64)a2 - *((_QWORD *)v11 - 1)) >> 5));
    v17 = *(_QWORD *)(v16 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v17 == v16 + 48 )
    {
      v19 = 0;
    }
    else
    {
      if ( !v17 )
        BUG();
      v18 = *(unsigned __int8 *)(v17 - 24);
      v19 = 0;
      v20 = v17 - 24;
      if ( (unsigned int)(v18 - 30) < 0xB )
        v19 = v20;
    }
    return sub_251BFD0(a1, v19, a3, a4, a5, a6, a7, 0);
  }
  if ( (_BYTE)v12 != 62 )
    goto LABEL_10;
  if ( a6 )
    goto LABEL_10;
  if ( *((_QWORD *)v11 - 4) == *a2 )
    goto LABEL_10;
  v37 = (unsigned __int8 *)a2[3];
  sub_250D230(&v41, (unsigned __int64)v11, 1, 0);
  v29 = sub_251BBC0(a1, v41, v42, a3, 2, 0, 1);
  v11 = v37;
  v30 = v29;
  if ( !v29
    || (v31 = *(__int64 (**)())(*(_QWORD *)v29 + 152LL), v31 == sub_2505DA0)
    || (result = ((__int64 (__fastcall *)(__int64, __int64))v31)(v30, v32), v11 = v37, !(_BYTE)result) )
  {
LABEL_10:
    sub_250D230(&v41, (unsigned __int64)v11, 1, 0);
    return sub_251C230(a1, (__int64 *)&v41, a3, a4, a5, a6, a7);
  }
  if ( a3 )
  {
    v40 = result;
    sub_250ED80(a1, v30, a3, a7);
    result = v40;
  }
  if ( (*(_BYTE *)(v30 + 96) & 2) == 0 )
    *a5 = 1;
  return result;
}
