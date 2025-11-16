// Function: sub_3394FC0
// Address: 0x3394fc0
//
__int64 __fastcall sub_3394FC0(__int64 a1, __int64 a2)
{
  __int16 v3; // r14
  __int64 v4; // r12
  __int64 v5; // rdx
  __int64 v6; // r13
  __int128 v7; // rax
  unsigned int v8; // r8d
  char v9; // dl
  int v10; // eax
  int v11; // ecx
  int v12; // ecx
  int v13; // ecx
  int v14; // ecx
  int v15; // ecx
  bool v16; // zf
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // rax
  int v21; // eax
  int v22; // edx
  __int64 v23; // r11
  __int64 v24; // rax
  int v25; // r9d
  unsigned int v26; // r8d
  __int64 v27; // rsi
  __int128 v28; // rax
  __int64 v29; // r12
  int v30; // edx
  int v31; // r13d
  _QWORD *v32; // rax
  __int64 v33; // rsi
  __int64 result; // rax
  __int128 v35; // [rsp-30h] [rbp-E0h]
  int v36; // [rsp+0h] [rbp-B0h]
  unsigned int v37; // [rsp+8h] [rbp-A8h]
  int v38; // [rsp+8h] [rbp-A8h]
  unsigned int v39; // [rsp+10h] [rbp-A0h]
  __int64 v40; // [rsp+10h] [rbp-A0h]
  int v41; // [rsp+10h] [rbp-A0h]
  __int64 *v42; // [rsp+18h] [rbp-98h]
  int v43; // [rsp+18h] [rbp-98h]
  __int128 v44; // [rsp+20h] [rbp-90h]
  __int64 v45; // [rsp+48h] [rbp-68h] BYREF
  __int64 v46; // [rsp+50h] [rbp-60h] BYREF
  int v47; // [rsp+58h] [rbp-58h]
  __int64 v48; // [rsp+60h] [rbp-50h] BYREF
  int v49; // [rsp+68h] [rbp-48h]
  __int64 v50; // [rsp+70h] [rbp-40h]

  v3 = *(_WORD *)(a2 + 2);
  v4 = sub_338B750(a1, *(_QWORD *)(a2 - 64));
  v6 = v5;
  *(_QWORD *)&v7 = sub_338B750(a1, *(_QWORD *)(a2 - 32));
  v44 = v7;
  v8 = sub_34B9180(v3 & 0x3F);
  v9 = *(_BYTE *)(a2 + 1) >> 1;
  if ( (v9 & 2) != 0 || (v10 = 0, (*(_BYTE *)(*(_QWORD *)(a1 + 856) + 864LL) & 4) != 0) )
  {
    v8 = sub_34B9190(v8);
    v9 = *(_BYTE *)(a2 + 1) >> 1;
    v10 = (16 * v9) & 0x20;
  }
  v39 = v8;
  if ( (v9 & 4) != 0 )
    v10 |= 0x40u;
  v11 = v10;
  if ( (v9 & 8) != 0 )
  {
    LOBYTE(v11) = v10 | 0x80;
    v10 = v11;
  }
  v12 = v10;
  if ( (v9 & 0x10) != 0 )
  {
    BYTE1(v12) = BYTE1(v10) | 1;
    v10 = v12;
  }
  v13 = v10;
  if ( (v9 & 0x20) != 0 )
  {
    BYTE1(v13) = BYTE1(v10) | 2;
    v10 = v13;
  }
  v14 = v10;
  if ( (v9 & 0x40) != 0 )
  {
    BYTE1(v14) = BYTE1(v10) | 4;
    v10 = v14;
  }
  v15 = v10;
  v16 = (v9 & 1) == 0;
  v17 = *(_QWORD *)(a1 + 864);
  if ( !v16 )
  {
    BYTE1(v15) = BYTE1(v10) | 8;
    v10 = v15;
  }
  v48 = *(_QWORD *)(a1 + 864);
  v49 = v10;
  v50 = *(_QWORD *)(v17 + 1024);
  *(_QWORD *)(v17 + 1024) = &v48;
  v18 = *(_QWORD *)(a1 + 864);
  v19 = *(_QWORD *)(v18 + 16);
  v42 = *(__int64 **)(a2 + 8);
  v20 = sub_2E79000(*(__int64 **)(v18 + 40));
  v21 = sub_2D5BAE0(v19, v20, v42, 0);
  v46 = 0;
  v23 = *(_QWORD *)(a1 + 864);
  v43 = v21;
  v24 = *(_QWORD *)a1;
  v25 = v22;
  v26 = v39;
  v16 = *(_QWORD *)a1 == 0;
  v47 = *(_DWORD *)(a1 + 848);
  if ( !v16 && &v46 != (__int64 *)(v24 + 48) )
  {
    v27 = *(_QWORD *)(v24 + 48);
    v46 = v27;
    if ( v27 )
    {
      v36 = v22;
      v37 = v39;
      v40 = v23;
      sub_B96E90((__int64)&v46, v27, 1);
      v25 = v36;
      v26 = v37;
      v23 = v40;
    }
  }
  v38 = v25;
  v41 = v23;
  *(_QWORD *)&v28 = sub_33ED040(v23, v26);
  *((_QWORD *)&v35 + 1) = v6;
  *(_QWORD *)&v35 = v4;
  v29 = sub_340F900(v41, 208, (unsigned int)&v46, v43, v38, v38, v35, v44, v28);
  v31 = v30;
  v45 = a2;
  v32 = sub_337DC20(a1 + 8, &v45);
  *v32 = v29;
  v33 = v46;
  *((_DWORD *)v32 + 2) = v31;
  if ( v33 )
    sub_B91220((__int64)&v46, v33);
  result = v48;
  *(_QWORD *)(v48 + 1024) = v50;
  return result;
}
