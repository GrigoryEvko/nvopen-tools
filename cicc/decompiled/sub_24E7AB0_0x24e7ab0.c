// Function: sub_24E7AB0
// Address: 0x24e7ab0
//
__int64 __fastcall sub_24E7AB0(__int64 a1, __int64 *a2)
{
  __int64 v4; // rdi
  __int16 v5; // dx
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r14
  unsigned __int64 v13; // rax
  int v14; // ecx
  _QWORD *v15; // rdx
  __int64 v16; // r13
  unsigned __int8 v17; // al
  unsigned int v18; // r14d
  _QWORD *v19; // rax
  __int64 v20; // r13
  unsigned __int64 v21; // r14
  _BYTE *v22; // r12
  __int64 v23; // rdx
  unsigned int v24; // esi
  unsigned __int64 v26; // rsi
  __int16 v27; // [rsp+Fh] [rbp-141h]
  unsigned __int8 v28; // [rsp+10h] [rbp-140h]
  _BYTE v29[32]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v30; // [rsp+50h] [rbp-100h]
  _QWORD v31[4]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v32; // [rsp+80h] [rbp-D0h]
  _BYTE *v33; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v34; // [rsp+98h] [rbp-B8h]
  _BYTE v35[32]; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v36; // [rsp+C0h] [rbp-90h]
  __int64 v37; // [rsp+C8h] [rbp-88h]
  __int64 v38; // [rsp+D0h] [rbp-80h]
  __int64 v39; // [rsp+D8h] [rbp-78h]
  void **v40; // [rsp+E0h] [rbp-70h]
  void **v41; // [rsp+E8h] [rbp-68h]
  __int64 v42; // [rsp+F0h] [rbp-60h]
  int v43; // [rsp+F8h] [rbp-58h]
  __int16 v44; // [rsp+FCh] [rbp-54h]
  char v45; // [rsp+FEh] [rbp-52h]
  __int64 v46; // [rsp+100h] [rbp-50h]
  __int64 v47; // [rsp+108h] [rbp-48h]
  void *v48; // [rsp+110h] [rbp-40h] BYREF
  void *v49; // [rsp+118h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 80LL);
  if ( v4 )
    v4 -= 24;
  v6 = sub_AA5030(v4, 1);
  if ( v6 )
    v27 = v5;
  else
    v27 = 0;
  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 80LL);
  if ( v7 )
    v7 -= 24;
  v41 = &v49;
  v39 = sub_AA48A0(v7);
  v40 = &v48;
  v33 = v35;
  v48 = &unk_49DA100;
  v34 = 0x200000000LL;
  v42 = 0;
  v49 = &unk_49DA0B0;
  v43 = 0;
  v44 = 512;
  LOWORD(v38) = v27;
  v45 = 7;
  v46 = 0;
  v47 = 0;
  v36 = v7;
  v37 = v6;
  if ( v6 != v7 + 48 )
  {
    v8 = v6 - 24;
    if ( !v6 )
      v8 = 0;
    v9 = *(_QWORD *)sub_B46C60(v8);
    v31[0] = v9;
    if ( v9 && (sub_B96E90((__int64)v31, v9, 1), (v12 = v31[0]) != 0) )
    {
      v13 = (unsigned __int64)v33;
      v14 = v34;
      v15 = &v33[16 * (unsigned int)v34];
      if ( v33 != (_BYTE *)v15 )
      {
        while ( *(_DWORD *)v13 )
        {
          v13 += 16LL;
          if ( v15 == (_QWORD *)v13 )
            goto LABEL_28;
        }
        *(_QWORD *)(v13 + 8) = v31[0];
        goto LABEL_17;
      }
LABEL_28:
      if ( (unsigned int)v34 >= (unsigned __int64)HIDWORD(v34) )
      {
        v26 = (unsigned int)v34 + 1LL;
        if ( HIDWORD(v34) < v26 )
        {
          sub_C8D5F0((__int64)&v33, v35, v26, 0x10u, v10, v11);
          v15 = &v33[16 * (unsigned int)v34];
        }
        *v15 = 0;
        v15[1] = v12;
        v12 = v31[0];
        LODWORD(v34) = v34 + 1;
      }
      else
      {
        if ( v15 )
        {
          *(_DWORD *)v15 = 0;
          v15[1] = v12;
          v14 = v34;
          v12 = v31[0];
        }
        LODWORD(v34) = v14 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v33, 0);
      v12 = v31[0];
    }
    if ( !v12 )
    {
      v7 = v36;
      goto LABEL_18;
    }
LABEL_17:
    sub_B91220((__int64)v31, v12);
    v7 = v36;
  }
LABEL_18:
  v30 = 257;
  v16 = sub_AA4E30(v7);
  v17 = sub_AE5260(v16, (__int64)a2);
  v18 = *(_DWORD *)(v16 + 4);
  v28 = v17;
  v32 = 257;
  v19 = sub_BD2C40(80, unk_3F10A14);
  v20 = (__int64)v19;
  if ( v19 )
    sub_B4CCA0((__int64)v19, a2, v18, 0, v28, (__int64)v31, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v41 + 2))(v41, v20, v29, v37, v38);
  v21 = (unsigned __int64)v33;
  v22 = &v33[16 * (unsigned int)v34];
  if ( v33 != v22 )
  {
    do
    {
      v23 = *(_QWORD *)(v21 + 8);
      v24 = *(_DWORD *)v21;
      v21 += 16LL;
      sub_B99FD0(v20, v24, v23);
    }
    while ( v22 != (_BYTE *)v21 );
  }
  *(_WORD *)(v20 + 2) |= 0x80u;
  **(_QWORD **)a1 = v20;
  nullsub_61();
  v48 = &unk_49DA100;
  nullsub_63();
  if ( v33 != v35 )
    _libc_free((unsigned __int64)v33);
  return v20;
}
