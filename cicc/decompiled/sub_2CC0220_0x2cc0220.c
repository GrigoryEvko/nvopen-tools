// Function: sub_2CC0220
// Address: 0x2cc0220
//
__int64 __fastcall sub_2CC0220(__int64 a1, unsigned __int8 *a2, char a3, __int64 a4, __int16 a5, __int64 a6)
{
  unsigned int v8; // r11d
  __int64 (__fastcall *v9)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v10; // rax
  _QWORD *v11; // r15
  __int64 v12; // r12
  _QWORD **v14; // rdx
  int v15; // ecx
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned int *v19; // rbx
  __int64 v20; // rdx
  unsigned int v21; // esi
  __int64 v22; // rax
  __int16 v23; // [rsp+8h] [rbp-138h]
  unsigned int *v24; // [rsp+8h] [rbp-138h]
  __int64 v25; // [rsp+18h] [rbp-128h]
  void *v26; // [rsp+20h] [rbp-120h] BYREF
  char v27; // [rsp+40h] [rbp-100h]
  char v28; // [rsp+41h] [rbp-FFh]
  _QWORD v29[4]; // [rsp+50h] [rbp-F0h] BYREF
  __int16 v30; // [rsp+70h] [rbp-D0h]
  unsigned int *v31; // [rsp+80h] [rbp-C0h] BYREF
  unsigned int v32; // [rsp+88h] [rbp-B8h]
  char v33; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v34; // [rsp+B8h] [rbp-88h]
  __int64 v35; // [rsp+C0h] [rbp-80h]
  __int64 v36; // [rsp+D0h] [rbp-70h]
  __int64 v37; // [rsp+D8h] [rbp-68h]
  void *v38; // [rsp+100h] [rbp-40h]

  if ( !a4 )
    BUG();
  sub_2412230((__int64)&v31, *(_QWORD *)(a4 + 16), a4, a5, 0, a6, 0, 0);
  v28 = 1;
  v26 = &unk_42D2000;
  v27 = 3;
  v8 = a3 == 0 ? 36 : 40;
  v9 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v36 + 56LL);
  if ( v9 != sub_928890 )
  {
    v22 = v9(v36, v8, (_BYTE *)a1, a2);
    v8 = a3 == 0 ? 36 : 40;
    v11 = (_QWORD *)v22;
LABEL_6:
    if ( v11 )
      goto LABEL_7;
    goto LABEL_10;
  }
  if ( *(_BYTE *)a1 <= 0x15u && *a2 <= 0x15u )
  {
    v10 = sub_AAB310(v8, (unsigned __int8 *)a1, a2);
    v8 = a3 == 0 ? 36 : 40;
    v11 = (_QWORD *)v10;
    goto LABEL_6;
  }
LABEL_10:
  v23 = v8;
  v30 = 257;
  v11 = sub_BD2C40(72, unk_3F10FD0);
  if ( v11 )
  {
    v14 = *(_QWORD ***)(a1 + 8);
    v15 = *((unsigned __int8 *)v14 + 8);
    if ( (unsigned int)(v15 - 17) > 1 )
    {
      v17 = sub_BCB2A0(*v14);
    }
    else
    {
      BYTE4(v25) = (_BYTE)v15 == 18;
      LODWORD(v25) = *((_DWORD *)v14 + 8);
      v16 = (__int64 *)sub_BCB2A0(*v14);
      v17 = sub_BCE1B0(v16, v25);
    }
    sub_B523C0((__int64)v11, v17, 53, v23, a1, (__int64)a2, (__int64)v29, 0, 0, 0);
  }
  (*(void (__fastcall **)(__int64, _QWORD *, void **, __int64, __int64))(*(_QWORD *)v37 + 16LL))(
    v37,
    v11,
    &v26,
    v34,
    v35);
  v18 = 4LL * v32;
  v19 = v31;
  v24 = &v31[v18];
  while ( v24 != v19 )
  {
    v20 = *((_QWORD *)v19 + 1);
    v21 = *v19;
    v19 += 4;
    sub_B99FD0((__int64)v11, v21, v20);
  }
LABEL_7:
  v29[0] = &unk_42D2000;
  v30 = 259;
  v12 = sub_B36550(&v31, (__int64)v11, (__int64)a2, a1, (__int64)v29, 0);
  nullsub_61();
  v38 = &unk_49DA100;
  nullsub_63();
  if ( v31 != (unsigned int *)&v33 )
    _libc_free((unsigned __int64)v31);
  return v12;
}
