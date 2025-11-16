// Function: sub_33AB2F0
// Address: 0x33ab2f0
//
__int64 __fastcall sub_33AB2F0(__int64 a1, __int64 a2, int a3)
{
  unsigned int v5; // r13d
  char v7; // al
  int v8; // r9d
  int v9; // edx
  int v10; // edx
  int v11; // edx
  int v12; // edx
  int v13; // edx
  __int128 v14; // rax
  __int64 v15; // r10
  __int64 v16; // rdx
  __int64 v17; // r11
  int v18; // r9d
  int v19; // ecx
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // r14
  int v25; // edx
  _QWORD *v26; // rax
  __int64 v27; // rsi
  __int128 v28; // [rsp-10h] [rbp-B0h]
  __int64 v29; // [rsp+0h] [rbp-A0h]
  __int64 v30; // [rsp+8h] [rbp-98h]
  int v31; // [rsp+10h] [rbp-90h]
  int v32; // [rsp+1Ch] [rbp-84h]
  int v33; // [rsp+20h] [rbp-80h]
  int v34; // [rsp+20h] [rbp-80h]
  int v35; // [rsp+28h] [rbp-78h]
  __int128 v36; // [rsp+30h] [rbp-70h]
  int v37; // [rsp+30h] [rbp-70h]
  __int64 v38; // [rsp+58h] [rbp-48h] BYREF
  __int64 v39; // [rsp+60h] [rbp-40h] BYREF
  int v40; // [rsp+68h] [rbp-38h]

  v5 = sub_B49E20(a2);
  if ( (_BYTE)v5 )
  {
    v7 = *(_BYTE *)(a2 + 1) >> 1;
    v8 = (16 * v7) & 0x20;
    if ( (v7 & 4) != 0 )
      v8 = (16 * v7) & 0x20 | 0x40;
    v9 = v8;
    if ( (v7 & 8) != 0 )
    {
      LOBYTE(v9) = v8 | 0x80;
      v8 = v9;
    }
    v10 = v8;
    if ( (v7 & 0x10) != 0 )
    {
      BYTE1(v10) = BYTE1(v8) | 1;
      v8 = v10;
    }
    v11 = v8;
    if ( (v7 & 0x20) != 0 )
    {
      BYTE1(v11) = BYTE1(v8) | 2;
      v8 = v11;
    }
    v12 = v8;
    if ( (v7 & 0x40) != 0 )
    {
      BYTE1(v12) = BYTE1(v8) | 4;
      v8 = v12;
    }
    v13 = v8;
    if ( (*(_BYTE *)(a2 + 1) & 2) != 0 )
    {
      BYTE1(v13) = BYTE1(v8) | 8;
      v8 = v13;
    }
    v33 = v8;
    *(_QWORD *)&v14 = sub_338B750(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    v36 = v14;
    v15 = sub_338B750(a1, *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
    v17 = v16;
    LODWORD(v16) = *(_DWORD *)(a1 + 848);
    v18 = v33;
    v19 = *(unsigned __int16 *)(*(_QWORD *)(v36 + 48) + 16LL * DWORD2(v36));
    v20 = *(_QWORD *)(*(_QWORD *)(v36 + 48) + 16LL * DWORD2(v36) + 8);
    v39 = 0;
    v21 = *(_QWORD *)(a1 + 864);
    v40 = v16;
    v35 = v21;
    v22 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      if ( &v39 != (__int64 *)(v22 + 48) )
      {
        v23 = *(_QWORD *)(v22 + 48);
        v39 = v23;
        if ( v23 )
        {
          v31 = v19;
          v29 = v15;
          v30 = v17;
          v32 = v33;
          v34 = v20;
          sub_B96E90((__int64)&v39, v23, 1);
          v19 = v31;
          LODWORD(v20) = v34;
          v15 = v29;
          v17 = v30;
          v18 = v32;
        }
      }
    }
    *((_QWORD *)&v28 + 1) = v17;
    *(_QWORD *)&v28 = v15;
    v24 = sub_3405C90(v35, a3, (unsigned int)&v39, v19, v20, v18, v36, v28);
    v37 = v25;
    v38 = a2;
    v26 = sub_337DC20(a1 + 8, &v38);
    *v26 = v24;
    v27 = v39;
    *((_DWORD *)v26 + 2) = v37;
    if ( v27 )
      sub_B91220((__int64)&v39, v27);
  }
  return v5;
}
