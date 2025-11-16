// Function: sub_11CB400
// Address: 0x11cb400
//
__int64 __fastcall sub_11CB400(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // r15
  __int64 *v8; // r14
  __int64 v9; // r8
  unsigned __int64 v10; // r15
  __int64 v11; // rdx
  unsigned __int64 v12; // rax
  unsigned __int8 *v13; // rdx
  unsigned __int64 v14; // r11
  unsigned __int8 *v15; // r10
  __int64 v16; // rsi
  __int64 v17; // rax
  unsigned __int8 *v18; // rax
  __int64 v20; // rax
  __int64 v21; // rcx
  unsigned int v22; // edi
  int *v23; // rdx
  int v24; // esi
  int v25; // edx
  int v26; // r10d
  __int64 *v27; // [rsp+0h] [rbp-90h]
  unsigned __int8 *v28; // [rsp+0h] [rbp-90h]
  __int64 v29; // [rsp+8h] [rbp-88h]
  unsigned __int64 v30; // [rsp+8h] [rbp-88h]
  __int64 v31; // [rsp+10h] [rbp-80h]
  unsigned __int8 *v33; // [rsp+18h] [rbp-78h]
  _QWORD v34[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD *v35; // [rsp+30h] [rbp-60h] BYREF
  __int64 v36; // [rsp+38h] [rbp-58h]
  _QWORD v37[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v38; // [rsp+50h] [rbp-40h]

  v4 = 0;
  v8 = (__int64 *)sub_AA4B30(*(_QWORD *)(a3 + 48));
  if ( !sub_11C99B0(v8, a4, 0x11Du) )
    return v4;
  v9 = sub_BCD140(*(_QWORD **)(a3 + 72), *(_DWORD *)(*a4 + 172));
  v10 = a4[5] & 0x20000000;
  if ( (a4[5] & 0x20000000) != 0 )
  {
    v31 = 0;
    v10 = 0;
    goto LABEL_7;
  }
  v11 = *a4;
  if ( (((int)*(unsigned __int8 *)(*a4 + 71) >> 2) & 3) == 0 )
  {
    v31 = 0;
    goto LABEL_7;
  }
  if ( (((int)*(unsigned __int8 *)(*a4 + 71) >> 2) & 3) != 3 )
  {
    v20 = *(unsigned int *)(v11 + 160);
    v21 = *(_QWORD *)(v11 + 144);
    if ( (_DWORD)v20 )
    {
      v22 = ((_WORD)v20 - 1) & 0x2931;
      v23 = (int *)(v21 + 40LL * (((_WORD)v20 - 1) & 0x2931));
      v24 = *v23;
      if ( *v23 == 285 )
      {
LABEL_14:
        v10 = *((_QWORD *)v23 + 2);
        v31 = *((_QWORD *)v23 + 1);
        goto LABEL_7;
      }
      v25 = 1;
      while ( v24 != -1 )
      {
        v26 = v25 + 1;
        v22 = (v20 - 1) & (v25 + v22);
        v23 = (int *)(v21 + 40LL * v22);
        v24 = *v23;
        if ( *v23 == 285 )
          goto LABEL_14;
        v25 = v26;
      }
    }
    v23 = (int *)(v21 + 40 * v20);
    goto LABEL_14;
  }
  v10 = qword_4977328[570];
  v31 = 61207443;
LABEL_7:
  v27 = (__int64 *)v9;
  v29 = *(_QWORD *)(a2 + 8);
  v37[0] = sub_BCE3C0(*(__int64 **)(a3 + 72), 0);
  v37[1] = v29;
  v35 = v37;
  v36 = 0x200000002LL;
  v12 = sub_BCF480(v27, v37, 2, 0);
  v14 = sub_11C96C0((__int64)v8, a4, 0x11Du, v12, 0);
  v15 = v13;
  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) == 14 )
  {
    v28 = v13;
    v30 = v14;
    sub_11C9500((__int64)v8, v31, v10, a4);
    v15 = v28;
    v14 = v30;
  }
  v16 = v14;
  v38 = 261;
  v35 = (_QWORD *)v31;
  v17 = a1;
  v36 = v10;
  v33 = v15;
  v34[0] = v17;
  v34[1] = a2;
  v4 = sub_921880((unsigned int **)a3, v14, (int)v15, (int)v34, 2, (__int64)&v35, 0);
  v18 = sub_BD3990(v33, v16);
  if ( !*v18 )
    *(_WORD *)(v4 + 2) = *(_WORD *)(v4 + 2) & 0xF003 | (4 * ((*((_WORD *)v18 + 1) >> 4) & 0x3FF));
  return v4;
}
