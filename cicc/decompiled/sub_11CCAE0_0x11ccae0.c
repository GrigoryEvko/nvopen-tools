// Function: sub_11CCAE0
// Address: 0x11ccae0
//
__int64 __fastcall sub_11CCAE0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // r15
  __int64 *v6; // r14
  __int64 *v7; // r8
  unsigned __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // rax
  unsigned __int8 *v12; // rdx
  unsigned __int8 *v13; // rax
  __int64 v15; // rax
  __int64 v16; // rcx
  unsigned int v17; // edi
  int *v18; // rdx
  int v19; // esi
  int v20; // edx
  int v21; // r10d
  unsigned __int64 v22; // [rsp+8h] [rbp-78h]
  unsigned __int8 *v23; // [rsp+10h] [rbp-70h]
  __int64 v24; // [rsp+18h] [rbp-68h] BYREF
  _QWORD v25[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v26; // [rsp+40h] [rbp-40h]

  v3 = 0;
  v24 = a1;
  v6 = (__int64 *)sub_AA4B30(*(_QWORD *)(a2 + 48));
  if ( !sub_11C99B0(v6, a3, 0x189u) )
    return v3;
  v7 = (__int64 *)sub_BCD140(*(_QWORD **)(a2 + 72), *(_DWORD *)(*a3 + 172));
  v8 = a3[7] & 0x200;
  if ( (a3[7] & 0x200) != 0 )
  {
    v8 = 0;
    v10 = 0;
    goto LABEL_7;
  }
  v9 = *a3;
  if ( (((int)*(unsigned __int8 *)(*a3 + 98) >> 2) & 3) != 0 )
  {
    if ( (((int)*(unsigned __int8 *)(*a3 + 98) >> 2) & 3) == 3 )
    {
      v10 = 61610682;
      v8 = qword_4977328[786];
      goto LABEL_7;
    }
    v15 = *(unsigned int *)(v9 + 160);
    v16 = *(_QWORD *)(v9 + 144);
    if ( (_DWORD)v15 )
    {
      v17 = ((_WORD)v15 - 1) & 0x38CD;
      v18 = (int *)(v16 + 40LL * (((_WORD)v15 - 1) & 0x38CD));
      v19 = *v18;
      if ( *v18 == 393 )
      {
LABEL_12:
        v10 = *((_QWORD *)v18 + 1);
        v8 = *((_QWORD *)v18 + 2);
        goto LABEL_7;
      }
      v20 = 1;
      while ( v19 != -1 )
      {
        v21 = v20 + 1;
        v17 = (v15 - 1) & (v20 + v17);
        v18 = (int *)(v16 + 40LL * v17);
        v19 = *v18;
        if ( *v18 == 393 )
          goto LABEL_12;
        v20 = v21;
      }
    }
    v18 = (int *)(v16 + 40 * v15);
    goto LABEL_12;
  }
  v10 = 0;
LABEL_7:
  v11 = sub_11CC840((__int64)v6, a3, 0x189u, 0, v7, (__int64)v7);
  v23 = v12;
  v22 = v11;
  sub_11C9500((__int64)v6, v10, v8, a3);
  v25[1] = v8;
  v26 = 261;
  v25[0] = v10;
  v3 = sub_921880((unsigned int **)a2, v22, (int)v23, (int)&v24, 1, (__int64)v25, 0);
  v13 = sub_BD3990(v23, v22);
  if ( !*v13 )
    *(_WORD *)(v3 + 2) = *(_WORD *)(v3 + 2) & 0xF003 | (4 * ((*((_WORD *)v13 + 1) >> 4) & 0x3FF));
  return v3;
}
