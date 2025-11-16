// Function: sub_11CD400
// Address: 0x11cd400
//
__int64 __fastcall sub_11CD400(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, int a5)
{
  __int64 v8; // rbx
  __int64 *v9; // r14
  unsigned __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // eax
  __int64 *v14; // rax
  __int64 v15; // rax
  unsigned __int8 *v16; // rdx
  unsigned __int8 *v17; // r15
  unsigned __int8 *v18; // rax
  __int64 v20; // rax
  __int64 v21; // rcx
  unsigned int v22; // edi
  int *v23; // rdx
  int v24; // esi
  int v25; // edx
  int v26; // r9d
  __int64 v27; // [rsp+10h] [rbp-90h]
  __int64 v28; // [rsp+18h] [rbp-88h]
  unsigned __int64 v29; // [rsp+18h] [rbp-88h]
  _QWORD v32[2]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v33[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v34; // [rsp+60h] [rbp-40h]

  v8 = 0;
  v9 = (__int64 *)sub_AA4B30(*(_QWORD *)(a3 + 48));
  if ( !sub_11C99B0(v9, a4, 0xC0u) )
    return v8;
  v10 = a4[4] & 1;
  if ( (a4[4] & 1) != 0 )
  {
    v27 = 0;
    v10 = 0;
    goto LABEL_7;
  }
  v11 = *a4;
  if ( (*(_BYTE *)(*a4 + 48) & 3) != 0 )
  {
    if ( (*(_BYTE *)(*a4 + 48) & 3) == 3 )
    {
      v10 = qword_4977328[384];
      v27 = 66179252;
      goto LABEL_7;
    }
    v20 = *(unsigned int *)(v11 + 160);
    v21 = *(_QWORD *)(v11 + 144);
    if ( (_DWORD)v20 )
    {
      v22 = ((_WORD)v20 - 1) & 0x1BC0;
      v23 = (int *)(v21 + 40LL * (((_WORD)v20 - 1) & 0x1BC0));
      v24 = *v23;
      if ( *v23 == 192 )
      {
LABEL_12:
        v10 = *((_QWORD *)v23 + 2);
        v27 = *((_QWORD *)v23 + 1);
        goto LABEL_7;
      }
      v25 = 1;
      while ( v24 != -1 )
      {
        v26 = v25 + 1;
        v22 = (v20 - 1) & (v25 + v22);
        v23 = (int *)(v21 + 40LL * v22);
        v24 = *v23;
        if ( *v23 == 192 )
          goto LABEL_12;
        v25 = v26;
      }
    }
    v23 = (int *)(v21 + 40 * v20);
    goto LABEL_12;
  }
  v27 = 0;
LABEL_7:
  v12 = sub_AA4B30(*(_QWORD *)(a3 + 48));
  v13 = sub_97FA80(*a4, v12);
  v28 = sub_BCD140(*(_QWORD **)(a3 + 72), v13);
  v14 = (__int64 *)sub_BCE3C0(*(__int64 **)(a3 + 72), a5);
  v15 = sub_11CCEE0((__int64)v9, a4, 0xC0u, 0, v14, v28, v28);
  v17 = v16;
  v29 = v15;
  sub_11C9500((__int64)v9, v27, v10, a4);
  v34 = 261;
  v32[0] = a1;
  v33[1] = v10;
  v33[0] = v27;
  v32[1] = a2;
  v8 = sub_921880((unsigned int **)a3, v29, (int)v17, (int)v32, 2, (__int64)v33, 0);
  v18 = sub_BD3990(v17, v29);
  if ( !*v18 )
    *(_WORD *)(v8 + 2) = *(_WORD *)(v8 + 2) & 0xF003 | (4 * ((*((_WORD *)v18 + 1) >> 4) & 0x3FF));
  return v8;
}
