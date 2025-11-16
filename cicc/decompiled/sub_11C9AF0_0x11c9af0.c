// Function: sub_11C9AF0
// Address: 0x11c9af0
//
__int64 __fastcall sub_11C9AF0(
        unsigned int a1,
        __int64 *a2,
        const void *a3,
        __int64 a4,
        int a5,
        int a6,
        __int64 a7,
        __int64 *a8,
        unsigned __int8 a9)
{
  __int64 v10; // rbx
  __int64 *v11; // r15
  unsigned __int64 v12; // r14
  __int64 v13; // rdi
  char *v14; // rbx
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int8 *v17; // rdx
  unsigned __int8 *v18; // r12
  unsigned __int8 *v19; // rax
  __int64 v21; // rax
  __int64 v22; // rsi
  unsigned int v23; // ecx
  int *v24; // rdx
  int v25; // edi
  int v26; // edx
  int v27; // r8d
  unsigned __int64 v33; // [rsp+28h] [rbp-68h]
  _QWORD v34[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v35; // [rsp+50h] [rbp-40h]

  v10 = 0;
  v11 = (__int64 *)sub_AA4B30(*(_QWORD *)(a7 + 48));
  if ( !sub_11C99B0(v11, a8, a1) )
    return v10;
  v12 = a8[((unsigned __int64)a1 >> 6) + 1] & (1LL << a1);
  if ( v12 )
  {
    v12 = 0;
    v14 = 0;
    goto LABEL_7;
  }
  v13 = *a8;
  if ( (((int)*(unsigned __int8 *)(*a8 + (a1 >> 2)) >> (2 * (a1 & 3))) & 3) != 0 )
  {
    if ( (((int)*(unsigned __int8 *)(*a8 + (a1 >> 2)) >> (2 * (a1 & 3))) & 3) == 3 )
    {
      v14 = (&off_4977320)[2 * a1];
      v12 = qword_4977328[2 * a1];
      goto LABEL_7;
    }
    v21 = *(unsigned int *)(v13 + 160);
    v22 = *(_QWORD *)(v13 + 144);
    if ( (_DWORD)v21 )
    {
      v23 = (v21 - 1) & (37 * a1);
      v24 = (int *)(v22 + 40LL * v23);
      v25 = *v24;
      if ( a1 == *v24 )
      {
LABEL_12:
        v14 = (char *)*((_QWORD *)v24 + 1);
        v12 = *((_QWORD *)v24 + 2);
        goto LABEL_7;
      }
      v26 = 1;
      while ( v25 != -1 )
      {
        v27 = v26 + 1;
        v23 = (v21 - 1) & (v26 + v23);
        v24 = (int *)(v22 + 40LL * v23);
        v25 = *v24;
        if ( a1 == *v24 )
          goto LABEL_12;
        v26 = v27;
      }
    }
    v24 = (int *)(v22 + 40 * v21);
    goto LABEL_12;
  }
  v14 = 0;
LABEL_7:
  v15 = sub_BCF480(a2, a3, a4, a9);
  v16 = sub_11C99A0((__int64)v11, a8, a1, v15);
  v18 = v17;
  v33 = v16;
  sub_11C9500((__int64)v11, (__int64)v14, v12, a8);
  v34[0] = v14;
  v35 = 261;
  v34[1] = v12;
  v10 = sub_921880((unsigned int **)a7, v33, (int)v18, a5, a6, (__int64)v34, 0);
  v19 = sub_BD3990(v18, v33);
  if ( !*v19 )
    *(_WORD *)(v10 + 2) = *(_WORD *)(v10 + 2) & 0xF003 | (4 * ((*((_WORD *)v19 + 1) >> 4) & 0x3FF));
  return v10;
}
