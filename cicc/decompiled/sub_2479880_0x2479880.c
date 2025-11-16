// Function: sub_2479880
// Address: 0x2479880
//
void __fastcall sub_2479880(__int64 *a1, unsigned __int8 *a2)
{
  __int64 *v3; // rdx
  unsigned __int8 *v4; // r15
  unsigned __int8 *v5; // rdx
  __int64 v6; // rbx
  __int64 **v7; // rax
  __int64 v8; // rsi
  _QWORD *v9; // rax
  _BYTE *v10; // rcx
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rax
  int v13; // r10d
  unsigned int v14; // r10d
  unsigned __int8 *v15; // r11
  __int64 (__fastcall *v16)(__int64, _QWORD, unsigned __int8 *, unsigned __int8 *, __int64, __int64); // rax
  __int64 v17; // rax
  _BYTE *v18; // rbx
  __int64 v19; // rax
  int v20; // r15d
  __int64 v21; // r15
  unsigned int *v22; // r15
  __int64 v23; // rdx
  unsigned int v24; // esi
  __int64 v25; // rax
  __int64 v26; // [rsp-10h] [rbp-170h]
  __int64 v27; // [rsp-8h] [rbp-168h]
  unsigned __int8 *v28; // [rsp+8h] [rbp-158h]
  unsigned __int8 *v29; // [rsp+8h] [rbp-158h]
  _BYTE *v30; // [rsp+18h] [rbp-148h]
  __int64 **v31; // [rsp+28h] [rbp-138h]
  unsigned int v32; // [rsp+28h] [rbp-138h]
  unsigned int *v33; // [rsp+28h] [rbp-138h]
  unsigned int v34; // [rsp+28h] [rbp-138h]
  int v35; // [rsp+38h] [rbp-128h]
  _BYTE v36[32]; // [rsp+40h] [rbp-120h] BYREF
  __int16 v37; // [rsp+60h] [rbp-100h]
  _BYTE v38[32]; // [rsp+70h] [rbp-F0h] BYREF
  __int16 v39; // [rsp+90h] [rbp-D0h]
  unsigned int *v40; // [rsp+A0h] [rbp-C0h] BYREF
  unsigned int v41; // [rsp+A8h] [rbp-B8h]
  char v42; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v43; // [rsp+D8h] [rbp-88h]
  __int64 v44; // [rsp+E0h] [rbp-80h]
  __int64 v45; // [rsp+F0h] [rbp-70h]
  __int64 v46; // [rsp+F8h] [rbp-68h]
  __int64 v47; // [rsp+100h] [rbp-60h]
  int v48; // [rsp+108h] [rbp-58h]
  void *v49; // [rsp+120h] [rbp-40h]

  sub_23D0AB0((__int64)&v40, (__int64)a2, 0, 0, 0);
  if ( (a2[7] & 0x40) != 0 )
    v3 = (__int64 *)*((_QWORD *)a2 - 1);
  else
    v3 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  v4 = (unsigned __int8 *)sub_246F3F0((__int64)a1, *v3);
  if ( (a2[7] & 0x40) != 0 )
    v5 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
  else
    v5 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  v6 = sub_246F3F0((__int64)a1, *((_QWORD *)v5 + 4));
  v39 = 257;
  v7 = *(__int64 ***)(v6 + 8);
  v37 = 257;
  v8 = *(_QWORD *)(v6 + 8);
  v31 = v7;
  v9 = sub_2463540(a1, v8);
  v10 = v9;
  if ( v9 )
    v10 = (_BYTE *)sub_AD6530((__int64)v9, v8);
  v11 = sub_92B530(&v40, 0x21u, v6, v10, (__int64)v36);
  v12 = sub_24633A0((__int64 *)&v40, 0x28u, v11, v31, (__int64)v38, 0, v35, 0);
  v13 = *a2;
  v37 = 257;
  v30 = (_BYTE *)v12;
  v14 = v13 - 29;
  v15 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
  v16 = *(__int64 (__fastcall **)(__int64, _QWORD, unsigned __int8 *, unsigned __int8 *, __int64, __int64))(*(_QWORD *)v45 + 16LL);
  if ( (char *)v16 != (char *)sub_9202E0 )
  {
    v29 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
    v34 = v14;
    v25 = v16(v45, v14, v4, v15, v26, v27);
    v15 = v29;
    v14 = v34;
    v18 = (_BYTE *)v25;
    goto LABEL_13;
  }
  if ( *v4 <= 0x15u && *v15 <= 0x15u )
  {
    v28 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
    v32 = v14;
    if ( (unsigned __int8)sub_AC47B0(v14) )
      v17 = sub_AD5570(v32, (__int64)v4, v28, 0, 0);
    else
      v17 = sub_AABE40(v32, v4, v28);
    v14 = v32;
    v15 = v28;
    v18 = (_BYTE *)v17;
LABEL_13:
    if ( v18 )
      goto LABEL_14;
  }
  v39 = 257;
  v18 = (_BYTE *)sub_B504D0(v14, (__int64)v4, (__int64)v15, (__int64)v38, 0, 0);
  if ( (unsigned __int8)sub_920620((__int64)v18) )
  {
    v20 = v48;
    if ( v47 )
      sub_B99FD0((__int64)v18, 3u, v47);
    sub_B45150((__int64)v18, v20);
  }
  (*(void (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v46 + 16LL))(v46, v18, v36, v43, v44);
  v21 = 4LL * v41;
  v33 = &v40[v21];
  if ( v40 != &v40[v21] )
  {
    v22 = v40;
    do
    {
      v23 = *((_QWORD *)v22 + 1);
      v24 = *v22;
      v22 += 4;
      sub_B99FD0((__int64)v18, v24, v23);
    }
    while ( v33 != v22 );
  }
LABEL_14:
  v39 = 257;
  v19 = sub_A82480(&v40, v18, v30, (__int64)v38);
  sub_246EF60((__int64)a1, (__int64)a2, v19);
  if ( *(_DWORD *)(a1[1] + 4) )
    sub_2477350((__int64)a1, (__int64)a2);
  nullsub_61();
  v49 = &unk_49DA100;
  nullsub_63();
  if ( v40 != (unsigned int *)&v42 )
    _libc_free((unsigned __int64)v40);
}
