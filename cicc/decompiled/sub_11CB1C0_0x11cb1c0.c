// Function: sub_11CB1C0
// Address: 0x11cb1c0
//
__int64 __fastcall sub_11CB1C0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // r15
  __int64 *v6; // r13
  __int64 *v7; // r14
  unsigned __int64 v8; // r15
  __int64 v9; // rdx
  int v10; // eax
  unsigned __int64 v11; // rax
  unsigned __int8 *v12; // rdx
  unsigned __int8 *v13; // r14
  unsigned __int8 *v14; // rax
  __int64 v16; // rax
  __int64 v17; // rcx
  unsigned int v18; // edi
  int *v19; // rdx
  int v20; // esi
  int v21; // edx
  int v22; // r9d
  __int64 v23; // [rsp+0h] [rbp-80h]
  __int64 v24; // [rsp+10h] [rbp-70h]
  __int64 v25; // [rsp+18h] [rbp-68h] BYREF
  __int64 v26; // [rsp+20h] [rbp-60h] BYREF
  __int64 v27; // [rsp+28h] [rbp-58h]
  __int64 v28; // [rsp+30h] [rbp-50h] BYREF
  __int16 v29; // [rsp+40h] [rbp-40h]

  v3 = 0;
  v25 = a1;
  v6 = (__int64 *)sub_AA4B30(*(_QWORD *)(a2 + 48));
  if ( !sub_11C99B0(v6, a3, 0x18Bu) )
    return v3;
  v7 = (__int64 *)sub_BCD140(*(_QWORD **)(a2 + 72), *(_DWORD *)(*a3 + 172));
  v8 = a3[7] & 0x800;
  if ( (a3[7] & 0x800) != 0 )
  {
    v23 = 0;
    v8 = 0;
    goto LABEL_7;
  }
  v9 = *a3;
  v10 = (int)*(unsigned __int8 *)(*a3 + 98) >> 6;
  if ( v10 )
  {
    if ( v10 == 3 )
    {
      v8 = qword_4977328[790];
      v23 = 61610727;
      goto LABEL_7;
    }
    v16 = *(unsigned int *)(v9 + 160);
    v17 = *(_QWORD *)(v9 + 144);
    if ( (_DWORD)v16 )
    {
      v18 = ((_WORD)v16 - 1) & 0x3917;
      v19 = (int *)(v17 + 40LL * (((_WORD)v16 - 1) & 0x3917));
      v20 = *v19;
      if ( *v19 == 395 )
      {
LABEL_12:
        v8 = *((_QWORD *)v19 + 2);
        v23 = *((_QWORD *)v19 + 1);
        goto LABEL_7;
      }
      v21 = 1;
      while ( v20 != -1 )
      {
        v22 = v21 + 1;
        v18 = (v16 - 1) & (v21 + v18);
        v19 = (int *)(v17 + 40LL * v18);
        v20 = *v19;
        if ( *v19 == 395 )
          goto LABEL_12;
        v21 = v22;
      }
    }
    v19 = (int *)(v17 + 40 * v16);
    goto LABEL_12;
  }
  v23 = 0;
LABEL_7:
  v28 = sub_BCE3C0(*(__int64 **)(a2 + 72), 0);
  v27 = 0x100000001LL;
  v11 = sub_BCF480(v7, &v28, 1, 0);
  v24 = sub_11C96C0((__int64)v6, a3, 0x18Bu, v11, 0);
  v13 = v12;
  sub_11C9500((__int64)v6, v23, v8, a3);
  v27 = v8;
  v29 = 261;
  v26 = v23;
  v3 = sub_921880((unsigned int **)a2, v24, (int)v13, (int)&v25, 1, (__int64)&v26, 0);
  v14 = sub_BD3990(v13, v24);
  if ( !*v14 )
    *(_WORD *)(v3 + 2) = *(_WORD *)(v3 + 2) & 0xF003 | (4 * ((*((_WORD *)v14 + 1) >> 4) & 0x3FF));
  return v3;
}
