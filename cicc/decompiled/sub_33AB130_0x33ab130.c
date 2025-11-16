// Function: sub_33AB130
// Address: 0x33ab130
//
__int64 __fastcall sub_33AB130(__int64 a1, __int64 a2, int a3)
{
  unsigned int v5; // r13d
  char v7; // al
  int v8; // r9d
  int v9; // edx
  int v10; // edx
  int v11; // edx
  int v12; // edx
  int v13; // edx
  __int64 v14; // rax
  int v15; // r9d
  __int64 v16; // r10
  __int64 v17; // rdx
  __int64 v18; // r11
  __int64 v19; // rax
  unsigned __int16 *v20; // rax
  int v21; // ecx
  __int64 v22; // r8
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // r14
  int v26; // edx
  _QWORD *v27; // rax
  __int64 v28; // rsi
  __int64 v29; // [rsp+0h] [rbp-90h]
  __int64 v30; // [rsp+8h] [rbp-88h]
  int v31; // [rsp+10h] [rbp-80h]
  int v32; // [rsp+1Ch] [rbp-74h]
  int v33; // [rsp+20h] [rbp-70h]
  int v34; // [rsp+20h] [rbp-70h]
  __int64 v35; // [rsp+28h] [rbp-68h]
  int v36; // [rsp+28h] [rbp-68h]
  __int64 v37; // [rsp+48h] [rbp-48h] BYREF
  __int64 v38; // [rsp+50h] [rbp-40h] BYREF
  int v39; // [rsp+58h] [rbp-38h]

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
    v14 = sub_338B750(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    v15 = v33;
    v16 = v14;
    v18 = v17;
    v35 = *(_QWORD *)(a1 + 864);
    v19 = (unsigned int)v17;
    LODWORD(v17) = *(_DWORD *)(a1 + 848);
    v20 = (unsigned __int16 *)(*(_QWORD *)(v16 + 48) + 16 * v19);
    v21 = *v20;
    v22 = *((_QWORD *)v20 + 1);
    v38 = 0;
    v23 = *(_QWORD *)a1;
    v39 = v17;
    if ( v23 )
    {
      if ( &v38 != (__int64 *)(v23 + 48) )
      {
        v24 = *(_QWORD *)(v23 + 48);
        v38 = v24;
        if ( v24 )
        {
          v31 = v21;
          v29 = v16;
          v30 = v18;
          v32 = v33;
          v34 = v22;
          sub_B96E90((__int64)&v38, v24, 1);
          v21 = v31;
          v16 = v29;
          v18 = v30;
          v15 = v32;
          LODWORD(v22) = v34;
        }
      }
    }
    v37 = a2;
    v25 = sub_33FA050(v35, a3, (unsigned int)&v38, v21, v22, v15, v16, v18);
    v36 = v26;
    v27 = sub_337DC20(a1 + 8, &v37);
    *v27 = v25;
    v28 = v38;
    *((_DWORD *)v27 + 2) = v36;
    if ( v28 )
      sub_B91220((__int64)&v38, v28);
  }
  return v5;
}
