// Function: sub_3144140
// Address: 0x3144140
//
void __fastcall sub_3144140(__int64 a1, float a2)
{
  char v3; // al
  __int64 v4; // rax
  __int64 v5; // rdi
  unsigned __int8 v6; // al
  __int64 v7; // rdx
  unsigned int v8; // eax
  unsigned int v9; // edx
  int v10; // esi
  _QWORD *v11; // rax
  __int64 v12; // rsi
  unsigned __int8 *v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // r9
  __int64 v17; // r8
  unsigned __int64 v18; // rax
  int v19; // ecx
  unsigned __int8 *v20; // rdx
  unsigned __int64 v21; // rsi
  __int64 v22; // rdx
  _QWORD *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  float v26; // xmm1_4
  unsigned __int64 v27; // rsi
  int v28; // edx
  int v29; // edx
  __int64 v30; // [rsp+0h] [rbp-E0h]
  __int64 v31; // [rsp+18h] [rbp-C8h] BYREF
  unsigned __int8 *v32; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v33; // [rsp+28h] [rbp-B8h]
  _BYTE v34[32]; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v35; // [rsp+50h] [rbp-90h]
  __int64 v36; // [rsp+58h] [rbp-88h]
  __int16 v37; // [rsp+60h] [rbp-80h]
  _QWORD *v38; // [rsp+68h] [rbp-78h]
  void **v39; // [rsp+70h] [rbp-70h]
  void **v40; // [rsp+78h] [rbp-68h]
  __int64 v41; // [rsp+80h] [rbp-60h]
  int v42; // [rsp+88h] [rbp-58h]
  __int16 v43; // [rsp+8Ch] [rbp-54h]
  char v44; // [rsp+8Eh] [rbp-52h]
  __int64 v45; // [rsp+90h] [rbp-50h]
  __int64 v46; // [rsp+98h] [rbp-48h]
  void *v47; // [rsp+A0h] [rbp-40h] BYREF
  void *v48; // [rsp+A8h] [rbp-38h] BYREF

  v3 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 != 85 )
  {
    if ( v3 != 34 && v3 != 40 )
      return;
    goto LABEL_9;
  }
  v4 = *(_QWORD *)(a1 - 32);
  if ( !v4 )
  {
LABEL_9:
    if ( *(_QWORD *)(a1 + 48) )
    {
      v5 = sub_B10CD0(a1 + 48);
      v6 = *(_BYTE *)(v5 - 16);
      v7 = (v6 & 2) != 0 ? *(_QWORD *)(v5 - 32) : v5 - 16 - 8LL * ((v6 >> 2) & 0xF);
      if ( **(_BYTE **)v7 == 20 )
      {
        v8 = *(_DWORD *)(*(_QWORD *)v7 + 4LL);
        if ( (v8 & 7) == 7 && (v8 & 0xFFFFFFF8) != 0 )
        {
          v9 = v8 >> 3;
          if ( (v8 & 0x10000000) != 0 )
          {
            v28 = (unsigned __int16)v8 >> 3;
            if ( a2 >= 1.0 )
              v29 = v8 & 0xC000000 | (8 * v28) | 0x3200007;
            else
              v29 = v8 & 0xC000000 | ((int)(float)(a2 * 100.0) << 19) | (8 * v28) | 7;
            v10 = v29 | v8 & 0x70000 | 0x10000000;
          }
          else if ( a2 < 1.0 )
          {
            v10 = ((int)(float)(a2 * 100.0) << 19) | v8 & 0xC000000 | (8 * (unsigned __int16)v9) | 7;
          }
          else
          {
            v10 = (8 * (unsigned __int16)v9) | v8 & 0xC000000 | 0x3200007;
          }
          v11 = sub_26BDBC0(v5, v10);
          sub_B10CB0(&v32, (__int64)v11);
          v12 = *(_QWORD *)(a1 + 48);
          if ( v12 )
            sub_B91220(a1 + 48, v12);
          v13 = v32;
          *(_QWORD *)(a1 + 48) = v32;
          if ( v13 )
            sub_B976B0((__int64)&v32, v13, a1 + 48);
        }
      }
    }
    return;
  }
  if ( *(_BYTE *)v4
    || *(_QWORD *)(v4 + 24) != *(_QWORD *)(a1 + 80)
    || (*(_BYTE *)(v4 + 33) & 0x20) == 0
    || *(_DWORD *)(v4 + 36) != 291 )
  {
    if ( !*(_BYTE *)v4 && *(_QWORD *)(v4 + 24) == *(_QWORD *)(a1 + 80) && (*(_BYTE *)(v4 + 33) & 0x20) != 0 )
      return;
    goto LABEL_9;
  }
  v38 = (_QWORD *)sub_BD5C60(a1);
  v39 = &v47;
  v32 = v34;
  v47 = &unk_49DA100;
  v33 = 0x200000000LL;
  v40 = &v48;
  v48 = &unk_49DA0B0;
  v14 = *(_QWORD *)(a1 + 40);
  v41 = 0;
  v35 = v14;
  v42 = 0;
  v43 = 512;
  v44 = 7;
  v45 = 0;
  v46 = 0;
  v36 = a1 + 24;
  v37 = 0;
  v15 = *(_QWORD *)sub_B46C60(a1);
  v31 = v15;
  if ( !v15 || (sub_B96E90((__int64)&v31, v15, 1), (v17 = v31) == 0) )
  {
    sub_93FB40((__int64)&v32, 0);
    v17 = v31;
    goto LABEL_44;
  }
  v18 = (unsigned __int64)v32;
  v19 = v33;
  v20 = &v32[16 * (unsigned int)v33];
  if ( v32 == v20 )
  {
LABEL_46:
    if ( (unsigned int)v33 >= (unsigned __int64)HIDWORD(v33) )
    {
      v27 = (unsigned int)v33 + 1LL;
      if ( HIDWORD(v33) < v27 )
      {
        v30 = v31;
        sub_C8D5F0((__int64)&v32, v34, v27, 0x10u, v31, v16);
        v17 = v30;
        v20 = &v32[16 * (unsigned int)v33];
      }
      *(_QWORD *)v20 = 0;
      *((_QWORD *)v20 + 1) = v17;
      v17 = v31;
      LODWORD(v33) = v33 + 1;
    }
    else
    {
      if ( v20 )
      {
        *(_DWORD *)v20 = 0;
        *((_QWORD *)v20 + 1) = v17;
        v17 = v31;
        v19 = v33;
      }
      LODWORD(v33) = v19 + 1;
    }
LABEL_44:
    if ( !v17 )
      goto LABEL_36;
    goto LABEL_35;
  }
  while ( *(_DWORD *)v18 )
  {
    v18 += 16LL;
    if ( v20 == (unsigned __int8 *)v18 )
      goto LABEL_46;
  }
  *(_QWORD *)(v18 + 8) = v31;
LABEL_35:
  sub_B91220((__int64)&v31, v17);
LABEL_36:
  v21 = -1;
  if ( a2 < 1.0 )
  {
    v26 = a2 * 1.8446744e19;
    if ( (float)(a2 * 1.8446744e19) >= 9.223372e18 )
      v21 = (unsigned int)(int)(float)(v26 - 9.223372e18) ^ 0x8000000000000000LL;
    else
      v21 = (unsigned int)(int)v26;
  }
  v22 = *(_QWORD *)(a1 + 32 * (3LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
  v23 = *(_QWORD **)(v22 + 24);
  if ( *(_DWORD *)(v22 + 32) > 0x40u )
    v23 = (_QWORD *)*v23;
  if ( (_QWORD *)v21 != v23 )
  {
    v24 = sub_BCB2E0(v38);
    v25 = sub_ACD640(v24, v21, 0);
    sub_BD2ED0(a1, *(_QWORD *)(a1 + 32 * (3LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))), v25);
  }
  nullsub_61();
  v47 = &unk_49DA100;
  nullsub_63();
  if ( v32 != v34 )
    _libc_free((unsigned __int64)v32);
}
