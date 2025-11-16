// Function: sub_17D60D0
// Address: 0x17d60d0
//
_QWORD *__fastcall sub_17D60D0(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  __int64 **v4; // r12
  __int64 v5; // rax
  __int64 v6; // rsi
  char v7; // al
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 *v10; // r14
  unsigned __int64 v11; // rcx
  _QWORD *v12; // rdi
  unsigned int v13; // r13d
  unsigned __int64 v14; // rbx
  int v15; // esi
  __int64 v16; // rbx
  unsigned __int64 v17; // r13
  _QWORD *v18; // rbx
  __int64 *v19; // rax
  _QWORD *v20; // r14
  __int64 v21; // rax
  __int64 *v22; // rax
  __int64 v23; // r12
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v27; // rbx
  __int128 v28; // rdi
  __int64 *v29; // rax
  _QWORD *v30; // rax
  __int64 v31; // rax
  unsigned int v32; // [rsp+Ch] [rbp-74h]
  __int64 v33; // [rsp+10h] [rbp-70h]
  unsigned __int64 v34; // [rsp+18h] [rbp-68h]
  __int64 v35; // [rsp+18h] [rbp-68h]
  __int64 v36; // [rsp+18h] [rbp-68h]
  unsigned int v37; // [rsp+20h] [rbp-60h]
  int v38; // [rsp+24h] [rbp-5Ch]
  unsigned __int64 v40; // [rsp+30h] [rbp-50h]
  _QWORD v42[7]; // [rsp+48h] [rbp-38h] BYREF

  v33 = sub_1632FA0(*(_QWORD *)(a1[1] + 40LL));
  v4 = (__int64 **)((*a2 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
  v40 = sub_1389B50(a2);
  if ( (__int64 **)v40 == v4 )
  {
    v23 = 0;
    goto LABEL_20;
  }
  v38 = 176;
  v37 = 48;
  v32 = 0;
  do
  {
    while ( 1 )
    {
      v10 = *v4;
      v11 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
      v34 = v11;
      v12 = (_QWORD *)(v11 + 56);
      v13 = *(_DWORD *)(*(_QWORD *)(v11 + 64) + 12LL) - 1;
      v14 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)&v4[3 * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF)] - v11) >> 3);
      v15 = -1431655765 * ((__int64)((__int64)&v4[3 * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF)] - v11) >> 3);
      if ( (*a2 & 4) != 0 )
      {
        if ( (unsigned __int8)sub_1560290(v12, v15, 6) )
          goto LABEL_17;
        v5 = *(_QWORD *)(v34 - 24);
        if ( *(_BYTE *)(v5 + 16) )
          goto LABEL_6;
        break;
      }
      if ( (unsigned __int8)sub_1560290(v12, v15, 6) )
        goto LABEL_17;
      v5 = *(_QWORD *)(v34 - 72);
      if ( !*(_BYTE *)(v5 + 16) )
        break;
LABEL_6:
      v6 = *v10;
      v7 = *(_BYTE *)(*v10 + 8);
      if ( v7 == 16 )
      {
        if ( (unsigned __int8)(*(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL) - 1) > 5u )
          goto LABEL_24;
LABEL_9:
        if ( v37 <= 0xAF )
        {
          v8 = sub_17CE5E0((__int64)a1, v6, (__int64 *)a3, v37);
          v37 += 16;
          v9 = v8;
          goto LABEL_11;
        }
        goto LABEL_24;
      }
      if ( (unsigned __int8)(v7 - 1) <= 5u || v7 == 9 )
        goto LABEL_9;
      if ( v7 == 11 )
      {
        v6 = *v10;
        if ( (unsigned int)sub_1643030(*v10) > 0x40 )
          goto LABEL_24;
      }
      else if ( v7 != 15 )
      {
        goto LABEL_24;
      }
      if ( v32 <= 0x2F )
      {
        v31 = sub_17CE5E0((__int64)a1, v6, (__int64 *)a3, v32);
        v32 += 8;
        v9 = v31;
LABEL_11:
        if ( v13 <= (unsigned int)v14 )
          goto LABEL_26;
        goto LABEL_12;
      }
LABEL_24:
      if ( v13 <= (unsigned int)v14 )
      {
        v27 = sub_12BE0A0(v33, v6) + 7;
        v9 = sub_17CE5E0((__int64)a1, *v10, (__int64 *)a3, v38);
        v38 += v27 & 0xFFFFFFF8;
LABEL_26:
        *((_QWORD *)&v28 + 1) = v10;
        v36 = v9;
        *(_QWORD *)&v28 = a1[3];
        v29 = sub_17D4DA0(v28);
        v30 = sub_12A8F50((__int64 *)a3, (__int64)v29, v36, 0);
        sub_15F9450((__int64)v30, 8u);
      }
LABEL_12:
      v4 += 3;
      if ( (__int64 **)v40 == v4 )
        goto LABEL_19;
    }
    v42[0] = *(_QWORD *)(v5 + 112);
    if ( !(unsigned __int8)sub_1560290(v42, v14, 6) )
      goto LABEL_6;
LABEL_17:
    if ( v13 > (unsigned int)v14 )
      goto LABEL_12;
    v4 += 3;
    v16 = **(_QWORD **)(*v10 + 16);
    v17 = sub_12BE0A0(v33, v16);
    v18 = (_QWORD *)sub_17CE5E0((__int64)a1, v16, (__int64 *)a3, v38);
    v38 += (v17 + 7) & 0xFFFFFFF8;
    v35 = a1[3];
    v19 = (__int64 *)sub_1643330(*(_QWORD **)(a3 + 24));
    v20 = (_QWORD *)sub_17CFB40(v35, (__int64)v10, (__int64 *)a3, v19, 8u);
    v21 = sub_1643360(*(_QWORD **)(a3 + 24));
    v22 = (__int64 *)sub_159C470(v21, v17, 0);
    sub_15E7430((__int64 *)a3, v18, 8u, v20, 8u, v22, 0, 0, 0, 0, 0);
  }
  while ( (__int64 **)v40 != v4 );
LABEL_19:
  v23 = (unsigned int)(v38 - 176);
LABEL_20:
  v24 = sub_1643360(*(_QWORD **)(a3 + 24));
  v25 = sub_159C470(v24, v23, 0);
  return sub_12A8F50((__int64 *)a3, v25, *(_QWORD *)(a1[2] + 232LL), 0);
}
