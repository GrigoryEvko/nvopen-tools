// Function: sub_247A9C0
// Address: 0x247a9c0
//
void __fastcall sub_247A9C0(__int64 *a1, __int64 a2, unsigned int a3)
{
  __int64 *v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rsi
  unsigned __int64 v8; // r10
  __int64 **v9; // rbx
  _BYTE *v10; // rax
  unsigned __int64 v11; // rax
  _BYTE *v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  _QWORD *v15; // r11
  unsigned __int64 v16; // r10
  __int64 v17; // rax
  unsigned int v18; // esi
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 **v22; // rax
  __int64 v23; // rax
  __int64 *v24; // rax
  __int64 **v25; // rax
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 **v28; // rax
  unsigned __int64 v29; // rax
  __int64 *v30; // rax
  __int64 v31; // [rsp+0h] [rbp-150h]
  unsigned __int64 v32; // [rsp+8h] [rbp-148h]
  unsigned __int64 v33; // [rsp+8h] [rbp-148h]
  unsigned __int64 v34; // [rsp+10h] [rbp-140h]
  unsigned __int64 v35; // [rsp+10h] [rbp-140h]
  unsigned __int64 v37; // [rsp+18h] [rbp-138h]
  __int64 v38; // [rsp+28h] [rbp-128h]
  _QWORD v39[4]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v40; // [rsp+50h] [rbp-100h]
  _QWORD v41[4]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v42; // [rsp+80h] [rbp-D0h]
  unsigned int *v43[2]; // [rsp+90h] [rbp-C0h] BYREF
  char v44; // [rsp+A0h] [rbp-B0h] BYREF
  void *v45; // [rsp+110h] [rbp-40h]

  sub_23D0AB0((__int64)v43, a2, 0, 0, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v4 = *(__int64 **)(a2 - 8);
  else
    v4 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v34 = sub_246F3F0((__int64)a1, *v4);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v5 = *(_QWORD *)(a2 - 8);
  else
    v5 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v6 = sub_246F3F0((__int64)a1, *(_QWORD *)(v5 + 32));
  v7 = a3;
  v8 = v6;
  if ( a3 )
  {
    v33 = v6;
    v30 = (__int64 *)sub_BCCE00(*(_QWORD **)(a1[1] + 72), a3);
    v9 = (__int64 **)sub_BCDA70(v30, 0x40 / a3);
    v42 = 257;
    v34 = sub_24633A0((__int64 *)v43, 0x31u, v34, v9, (__int64)v41, 0, v39[0], 0);
    v42 = 257;
    v7 = 49;
    v8 = sub_24633A0((__int64 *)v43, 0x31u, v33, v9, (__int64)v41, 0, v39[0], 0);
  }
  else
  {
    v9 = *(__int64 ***)(v34 + 8);
  }
  v31 = v8;
  v42 = 257;
  v40 = 257;
  v10 = (_BYTE *)sub_AD6530((__int64)v9, v7);
  v11 = sub_92B530(v43, 0x21u, v34, v10, (__int64)v39);
  v35 = sub_24633A0((__int64 *)v43, 0x28u, v11, v9, (__int64)v41, 0, v38, 0);
  v42 = 257;
  v40 = 257;
  v12 = (_BYTE *)sub_AD6530((__int64)v9, 40);
  v13 = sub_92B530(v43, 0x21u, v31, v12, (__int64)v39);
  v14 = sub_24633A0((__int64 *)v43, 0x28u, v13, v9, (__int64)v41, 0, v38, 0);
  v15 = v39;
  v16 = v14;
  if ( a3 )
  {
    v23 = a1[1];
    v32 = v16;
    v42 = 257;
    v24 = (__int64 *)sub_BCCE00(*(_QWORD **)(v23 + 72), 0x40u);
    v25 = (__int64 **)sub_BCDA70(v24, 1);
    v35 = sub_24633A0((__int64 *)v43, 0x31u, v35, v25, (__int64)v41, 0, v39[0], 0);
    v26 = a1[1];
    v42 = 257;
    v27 = (__int64 *)sub_BCCE00(*(_QWORD **)(v26 + 72), 0x40u);
    v28 = (__int64 **)sub_BCDA70(v27, 1);
    v29 = sub_24633A0((__int64 *)v43, 0x31u, v32, v28, (__int64)v41, 0, v39[0], 0);
    LODWORD(v15) = (unsigned int)v39;
    v16 = v29;
  }
  v41[0] = "_msprop_vector_pack";
  v42 = 259;
  v39[0] = v35;
  v17 = *(_QWORD *)(a2 - 32);
  BYTE4(v38) = 0;
  v39[1] = v16;
  if ( !v17 || *(_BYTE *)v17 || *(_QWORD *)(v17 + 24) != *(_QWORD *)(a2 + 80) )
    BUG();
  v18 = *(_DWORD *)(v17 + 36);
  if ( v18 != 15565 )
  {
    if ( v18 <= 0x3CCD )
    {
      if ( v18 == 14715 )
      {
        v18 = 14713;
      }
      else if ( v18 <= 0x397B )
      {
        if ( v18 != 14713 && v18 != 14714 )
          goto LABEL_41;
      }
      else
      {
        if ( v18 != 14716 )
          goto LABEL_41;
        v18 = 14714;
      }
    }
    else if ( v18 > 0x3D81 )
    {
      if ( v18 != 15791 )
        goto LABEL_41;
      v18 = 15743;
    }
    else if ( v18 > 0x3D7F )
    {
      v18 = 15744;
    }
    else
    {
      if ( v18 <= 0x3CCF )
      {
        v18 = 15566;
        goto LABEL_18;
      }
      if ( v18 != 15743 )
LABEL_41:
        BUG();
    }
  }
LABEL_18:
  v19 = sub_B33D10((__int64)v43, v18, 0, 0, (int)v15, 2, v38, (__int64)v41);
  v20 = v19;
  if ( a3 )
  {
    v21 = *(_QWORD *)(a2 + 8);
    v37 = v19;
    v42 = 257;
    v22 = (__int64 **)sub_2463540(a1, v21);
    v20 = sub_24633A0((__int64 *)v43, 0x31u, v37, v22, (__int64)v41, 0, v39[0], 0);
  }
  sub_246EF60((__int64)a1, a2, v20);
  if ( *(_DWORD *)(a1[1] + 4) )
    sub_2477350((__int64)a1, a2);
  nullsub_61();
  v45 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v43[0] != &v44 )
    _libc_free((unsigned __int64)v43[0]);
}
