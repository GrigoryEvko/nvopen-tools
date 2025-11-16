// Function: sub_383CE80
// Address: 0x383ce80
//
__int64 *__fastcall sub_383CE80(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  _QWORD *v4; // r14
  __int64 v5; // r13
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // r10
  __int64 v8; // r11
  unsigned __int16 *v9; // rax
  __int64 v10; // rsi
  unsigned __int64 v11; // rcx
  unsigned int v12; // r15d
  __int64 v13; // rax
  unsigned int v14; // edx
  unsigned int v15; // ebx
  unsigned __int64 v16; // rdx
  __int128 v17; // rax
  unsigned __int8 *v18; // rax
  __int64 v19; // rdx
  unsigned __int8 *v21; // rax
  __int64 v22; // rdx
  __int128 v23; // [rsp-20h] [rbp-A0h]
  __int64 v24; // [rsp+8h] [rbp-78h]
  __int64 v25; // [rsp+18h] [rbp-68h]
  unsigned __int64 v26; // [rsp+20h] [rbp-60h]
  __int64 v27; // [rsp+20h] [rbp-60h]
  __int64 v28; // [rsp+28h] [rbp-58h]
  unsigned __int64 v29; // [rsp+30h] [rbp-50h]
  _QWORD *v30; // [rsp+30h] [rbp-50h]
  unsigned __int8 *v31; // [rsp+30h] [rbp-50h]
  __int64 v32; // [rsp+38h] [rbp-48h]
  __int64 v33; // [rsp+40h] [rbp-40h] BYREF
  int v34; // [rsp+48h] [rbp-38h]

  v2 = a1;
  v4 = *(_QWORD **)(a1 + 8);
  v5 = *(_QWORD *)(a2 + 40);
  if ( *(_DWORD *)(a2 + 24) == 455 )
  {
    v21 = sub_383B380(a1, *(_QWORD *)v5, *(_QWORD *)(v5 + 8));
    return sub_33EC3B0(
             v4,
             (__int64 *)a2,
             (__int64)v21,
             v22,
             *(_QWORD *)(v5 + 40),
             *(_QWORD *)(v5 + 48),
             *(_OWORD *)(v5 + 80));
  }
  else
  {
    v6 = *(_QWORD *)v5;
    v7 = *(_QWORD *)v5;
    v8 = *(_QWORD *)(v5 + 8);
    v9 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v5 + 48LL) + 16LL * *(unsigned int *)(v5 + 8));
    v10 = *(_QWORD *)(*(_QWORD *)v5 + 80LL);
    v11 = *((_QWORD *)v9 + 1);
    v12 = *v9;
    v33 = v10;
    v29 = v11;
    if ( v10 )
    {
      v26 = v7;
      v28 = v8;
      sub_B96E90((__int64)&v33, v10, 1);
      v2 = a1;
      v7 = v26;
      v8 = v28;
    }
    v24 = v8;
    v25 = v2;
    v34 = *(_DWORD *)(v6 + 72);
    v13 = sub_37AE0F0(v2, v7, v8);
    v15 = v14;
    v16 = v29;
    v27 = v13;
    v30 = *(_QWORD **)(v25 + 8);
    *(_QWORD *)&v17 = sub_33F7D60(v30, v12, v16);
    *((_QWORD *)&v23 + 1) = v15 | v24 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v23 = v27;
    v18 = sub_3406EB0(
            v30,
            0xDEu,
            (__int64)&v33,
            *(unsigned __int16 *)(*(_QWORD *)(v27 + 48) + 16LL * v15),
            *(_QWORD *)(*(_QWORD *)(v27 + 48) + 16LL * v15 + 8),
            0xFFFFFFFF00000000LL,
            v23,
            v17);
    if ( v33 )
    {
      v31 = v18;
      v32 = v19;
      sub_B91220((__int64)&v33, v33);
      v18 = v31;
      v19 = v32;
    }
    return sub_33EBEE0(v4, (__int64 *)a2, (__int64)v18, v19);
  }
}
