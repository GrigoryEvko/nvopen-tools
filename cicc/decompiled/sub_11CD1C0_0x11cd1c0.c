// Function: sub_11CD1C0
// Address: 0x11cd1c0
//
__int64 __fastcall sub_11CD1C0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // r15
  __int64 *v8; // r14
  __int64 *v9; // r8
  unsigned __int64 v10; // r15
  __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // rax
  unsigned __int8 *v14; // rdx
  unsigned __int64 v15; // r11
  unsigned __int8 *v16; // r10
  __int64 v17; // rsi
  __int64 v18; // rax
  unsigned __int8 *v19; // rax
  __int64 v21; // rax
  __int64 v22; // rcx
  unsigned int v23; // edi
  int *v24; // rdx
  int v25; // esi
  int v26; // edx
  int v27; // r10d
  unsigned __int8 *v28; // [rsp+0h] [rbp-90h]
  unsigned __int64 v29; // [rsp+8h] [rbp-88h]
  __int64 v30; // [rsp+10h] [rbp-80h]
  unsigned __int8 *v32; // [rsp+18h] [rbp-78h]
  _QWORD v33[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v34[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v35; // [rsp+50h] [rbp-40h]

  v4 = 0;
  v8 = (__int64 *)sub_AA4B30(*(_QWORD *)(a3 + 48));
  if ( !sub_11C99B0(v8, a4, 0x11Bu) )
    return v4;
  v9 = (__int64 *)sub_BCD140(*(_QWORD **)(a3 + 72), *(_DWORD *)(*a4 + 172));
  v10 = a4[5] & 0x8000000;
  if ( (a4[5] & 0x8000000) != 0 )
  {
    v30 = 0;
    v10 = 0;
    goto LABEL_7;
  }
  v11 = *a4;
  v12 = (int)*(unsigned __int8 *)(*a4 + 70) >> 6;
  if ( !v12 )
  {
    v30 = 0;
    goto LABEL_7;
  }
  if ( v12 != 3 )
  {
    v21 = *(unsigned int *)(v11 + 160);
    v22 = *(_QWORD *)(v11 + 144);
    if ( (_DWORD)v21 )
    {
      v23 = ((_WORD)v21 - 1) & 0x28E7;
      v24 = (int *)(v22 + 40LL * (((_WORD)v21 - 1) & 0x28E7));
      v25 = *v24;
      if ( *v24 == 283 )
      {
LABEL_14:
        v10 = *((_QWORD *)v24 + 2);
        v30 = *((_QWORD *)v24 + 1);
        goto LABEL_7;
      }
      v26 = 1;
      while ( v25 != -1 )
      {
        v27 = v26 + 1;
        v23 = (v21 - 1) & (v26 + v23);
        v24 = (int *)(v22 + 40LL * v23);
        v25 = *v24;
        if ( *v24 == 283 )
          goto LABEL_14;
        v26 = v27;
      }
    }
    v24 = (int *)(v22 + 40 * v21);
    goto LABEL_14;
  }
  v10 = qword_4977328[566];
  v30 = 61207402;
LABEL_7:
  v13 = sub_11CCEE0((__int64)v8, a4, 0x11Bu, 0, v9, (__int64)v9, *(_QWORD *)(a2 + 8));
  v15 = v13;
  v16 = v14;
  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) == 14 )
  {
    v28 = v14;
    v29 = v13;
    sub_11C9500((__int64)v8, v30, v10, a4);
    v16 = v28;
    v15 = v29;
  }
  v17 = v15;
  v35 = 261;
  v34[0] = v30;
  v18 = a1;
  v34[1] = v10;
  v32 = v16;
  v33[0] = v18;
  v33[1] = a2;
  v4 = sub_921880((unsigned int **)a3, v15, (int)v16, (int)v33, 2, (__int64)v34, 0);
  v19 = sub_BD3990(v32, v17);
  if ( !*v19 )
    *(_WORD *)(v4 + 2) = *(_WORD *)(v4 + 2) & 0xF003 | (4 * ((*((_WORD *)v19 + 1) >> 4) & 0x3FF));
  return v4;
}
