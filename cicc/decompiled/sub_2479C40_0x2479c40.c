// Function: sub_2479C40
// Address: 0x2479c40
//
void __fastcall sub_2479C40(__int64 *a1, __int64 a2, char a3)
{
  __int64 *v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 **v8; // rbx
  _QWORD *v9; // rax
  __int64 v10; // rdx
  _BYTE *v11; // rcx
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  _BYTE *v14; // rbx
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // r11
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 **v23; // rax
  _BYTE *v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // rbx
  __int64 v29; // rax
  __int64 v30; // rsi
  _QWORD *v31; // rax
  __int64 v32; // rdx
  _BYTE *v33; // rcx
  __int64 v34; // rax
  unsigned __int64 v35; // rax
  __int64 v36; // [rsp+10h] [rbp-140h]
  __int64 v37; // [rsp+10h] [rbp-140h]
  unsigned __int64 v38; // [rsp+10h] [rbp-140h]
  unsigned __int64 v39; // [rsp+10h] [rbp-140h]
  unsigned __int64 v40; // [rsp+18h] [rbp-138h]
  unsigned __int64 v41; // [rsp+18h] [rbp-138h]
  _QWORD v42[2]; // [rsp+20h] [rbp-130h] BYREF
  int v43[8]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v44; // [rsp+50h] [rbp-100h]
  _BYTE v45[32]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v46; // [rsp+80h] [rbp-D0h]
  unsigned int *v47[2]; // [rsp+90h] [rbp-C0h] BYREF
  char v48; // [rsp+A0h] [rbp-B0h] BYREF
  _QWORD *v49; // [rsp+D8h] [rbp-78h]
  void *v50; // [rsp+110h] [rbp-40h]

  sub_23D0AB0((__int64)v47, a2, 0, 0, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v5 = *(__int64 **)(a2 - 8);
  else
    v5 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v40 = sub_246F3F0((__int64)a1, *v5);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v6 = *(_QWORD *)(a2 - 8);
  else
    v6 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v7 = sub_246F3F0((__int64)a1, *(_QWORD *)(v6 + 32));
  if ( a3 )
  {
    v8 = *(__int64 ***)(v7 + 8);
    v36 = v7;
    v46 = 257;
    v9 = sub_2463540(a1, (__int64)v8);
    v10 = v36;
    v11 = v9;
    if ( v9 )
    {
      v12 = sub_AD6530((__int64)v9, (__int64)v8);
      v10 = v36;
      v11 = (_BYTE *)v12;
    }
    v13 = sub_92B530(v47, 0x21u, v10, v11, (__int64)v45);
    v46 = 257;
    v14 = (_BYTE *)sub_24633A0((__int64 *)v47, 0x28u, v13, v8, (__int64)v45, 0, v43[0], 0);
  }
  else
  {
    v38 = v7;
    v26 = sub_2463540(a1, *(_QWORD *)(a2 + 8));
    v27 = v38;
    v28 = (__int64)v26;
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v38 + 8) + 8LL) - 17 <= 1 )
    {
      v29 = sub_BCB2E0(v49);
      v27 = sub_2464970(a1, v47, v38, v29, 1);
    }
    v39 = v27;
    v46 = 257;
    v30 = *(_QWORD *)(v27 + 8);
    v31 = sub_2463540(a1, v30);
    v32 = v39;
    v33 = v31;
    if ( v31 )
    {
      v34 = sub_AD6530((__int64)v31, v30);
      v32 = v39;
      v33 = (_BYTE *)v34;
    }
    v35 = sub_92B530(v47, 0x21u, v32, v33, (__int64)v45);
    v14 = (_BYTE *)sub_2464970(a1, v47, v35, v28, 1);
  }
  v15 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v16 = 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v44 = 257;
  v17 = *(_QWORD *)(a2 + v16);
  v46 = 257;
  v37 = v17;
  v18 = sub_24633A0((__int64 *)v47, 0x31u, v40, *(__int64 ***)(v15 + 8), (__int64)v43, 0, v42[0], 0);
  v19 = *(_QWORD *)(a2 - 32);
  v20 = *(_QWORD *)(a2 + 80);
  v42[1] = v37;
  v42[0] = v18;
  v21 = sub_921880(v47, v20, v19, (int)v42, 2, (__int64)v45, 0);
  v22 = *(_QWORD *)(a2 + 8);
  v46 = 257;
  v41 = v21;
  v23 = (__int64 **)sub_2463540(a1, v22);
  v24 = (_BYTE *)sub_24633A0((__int64 *)v47, 0x31u, v41, v23, (__int64)v45, 0, v43[0], 0);
  v46 = 257;
  v25 = sub_A82480(v47, v24, v14, (__int64)v45);
  sub_246EF60((__int64)a1, a2, v25);
  if ( *(_DWORD *)(a1[1] + 4) )
    sub_2477350((__int64)a1, a2);
  nullsub_61();
  v50 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v47[0] != &v48 )
    _libc_free((unsigned __int64)v47[0]);
}
