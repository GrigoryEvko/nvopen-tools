// Function: sub_33949A0
// Address: 0x33949a0
//
void __fastcall sub_33949A0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int64 v4; // r14
  __int64 v5; // rdx
  __int64 v6; // r15
  __int64 v7; // rdx
  __int64 v8; // r10
  unsigned __int8 v9; // al
  __int64 v10; // rdx
  __int64 v11; // r11
  int v12; // r9d
  int v13; // edx
  int v14; // ecx
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // r14
  int v19; // edx
  int v20; // r15d
  _QWORD *v21; // rax
  __int64 v22; // rsi
  int v23; // edx
  __int128 v24; // [rsp-20h] [rbp-B0h]
  __int128 v25; // [rsp-10h] [rbp-A0h]
  __int64 v26; // [rsp+0h] [rbp-90h]
  __int64 v27; // [rsp+8h] [rbp-88h]
  int v28; // [rsp+10h] [rbp-80h]
  int v29; // [rsp+1Ch] [rbp-74h]
  int v30; // [rsp+20h] [rbp-70h]
  __int64 v31; // [rsp+28h] [rbp-68h]
  __int64 v32; // [rsp+48h] [rbp-48h] BYREF
  __int64 v33; // [rsp+50h] [rbp-40h] BYREF
  int v34; // [rsp+58h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(__int64 **)(a2 - 8);
  else
    v3 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v4 = sub_338B750(a1, *v3);
  v6 = v5;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v7 = *(_QWORD *)(a2 - 8);
  else
    v7 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v8 = sub_338B750(a1, *(_QWORD *)(v7 + 32));
  v9 = *(_BYTE *)a2;
  v11 = v10;
  if ( *(_BYTE *)a2 <= 0x1Cu )
  {
    v12 = 0;
    if ( v9 == 5 )
    {
      v23 = *(unsigned __int16 *)(a2 + 2);
      if ( (unsigned int)(v23 - 19) <= 1 || (unsigned __int16)(v23 - 26) <= 1u )
        goto LABEL_8;
    }
  }
  else if ( (unsigned int)v9 - 48 <= 1 || (v12 = 0, (unsigned __int8)(v9 - 55) <= 1u) )
  {
LABEL_8:
    v12 = 4 * ((*(_BYTE *)(a2 + 1) & 2) != 0);
  }
  v13 = *(_DWORD *)(a1 + 848);
  v31 = *(_QWORD *)(a1 + 864);
  v14 = *(unsigned __int16 *)(*(_QWORD *)(v4 + 48) + 16LL * (unsigned int)v6);
  v15 = *(_QWORD *)(*(_QWORD *)(v4 + 48) + 16LL * (unsigned int)v6 + 8);
  v33 = 0;
  v16 = *(_QWORD *)a1;
  v34 = v13;
  if ( v16 )
  {
    if ( &v33 != (__int64 *)(v16 + 48) )
    {
      v17 = *(_QWORD *)(v16 + 48);
      v33 = v17;
      if ( v17 )
      {
        v28 = v14;
        v26 = v8;
        v27 = v11;
        v29 = v12;
        v30 = v15;
        sub_B96E90((__int64)&v33, v17, 1);
        v14 = v28;
        v8 = v26;
        v11 = v27;
        v12 = v29;
        LODWORD(v15) = v30;
      }
    }
  }
  *((_QWORD *)&v25 + 1) = v11;
  *(_QWORD *)&v25 = v8;
  *((_QWORD *)&v24 + 1) = v6;
  *(_QWORD *)&v24 = v4;
  v18 = sub_3405C90(v31, 59, (unsigned int)&v33, v14, v15, v12, v24, v25);
  v20 = v19;
  v32 = a2;
  v21 = sub_337DC20(a1 + 8, &v32);
  *v21 = v18;
  v22 = v33;
  *((_DWORD *)v21 + 2) = v20;
  if ( v22 )
    sub_B91220((__int64)&v33, v22);
}
