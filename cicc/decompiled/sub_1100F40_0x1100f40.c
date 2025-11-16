// Function: sub_1100F40
// Address: 0x1100f40
//
unsigned __int8 *__fastcall sub_1100F40(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v6; // rdx
  __int64 v7; // r14
  int v9; // r13d
  __int64 v10; // rbx
  __int64 *v11; // rax
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 *v15; // r12
  __int64 v16; // r15
  __int64 v17; // r13
  int v18; // eax
  int v19; // ecx
  __int64 v20; // rdx
  __int64 *v21; // rax
  __int64 *v22; // r12
  __int64 v23; // rdx
  __int64 v24; // r15
  const char *v25; // rax
  __int64 *v26; // r15
  __int64 v27; // rdx
  _QWORD *v28; // r12
  __int64 v29; // r11
  unsigned int v30; // ecx
  unsigned __int8 v31; // al
  __int64 v32; // rsi
  __int64 *v33; // rdi
  __int64 *v34; // rax
  __int64 v35; // rsi
  int v36; // edx
  char v37; // al
  __int64 v38; // rax
  __int64 v39; // rdx
  int v40; // r15d
  __int64 v41; // rbx
  __int64 v42; // r12
  __int64 v43; // rdx
  unsigned int v44; // esi
  int v45; // [rsp+Ch] [rbp-D4h]
  __int64 v46; // [rsp+10h] [rbp-D0h]
  __int64 v47; // [rsp+18h] [rbp-C8h]
  __int64 v48; // [rsp+28h] [rbp-B8h]
  _QWORD v49[4]; // [rsp+30h] [rbp-B0h] BYREF
  __int16 v50; // [rsp+50h] [rbp-90h]
  __int64 *v51; // [rsp+60h] [rbp-80h] BYREF
  __int64 v52; // [rsp+68h] [rbp-78h]
  _BYTE v53[16]; // [rsp+70h] [rbp-70h] BYREF
  __int16 v54; // [rsp+80h] [rbp-60h]

  v6 = *(_QWORD *)(a2 + 8);
  v7 = *(_QWORD *)(a2 - 32);
  if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 <= 1 )
    v6 = **(_QWORD **)(v6 + 16);
  if ( *(_BYTE *)v7 != 63 )
    return sub_11005E0(a1, (unsigned __int8 *)a2, v6, a4, a5, a6);
  v9 = *(_DWORD *)(v6 + 8) >> 8;
  v10 = *(_QWORD *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF));
  v11 = (__int64 *)sub_BD5C60(*(_QWORD *)(a2 - 32));
  v12 = sub_BCE3C0(v11, v9);
  v15 = (__int64 *)a1[4];
  v50 = 257;
  v16 = v12;
  if ( v12 == *(_QWORD *)(v10 + 8) )
  {
    v17 = v10;
  }
  else
  {
    v17 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v15[10] + 120LL))(
            v15[10],
            50,
            v10,
            v12);
    if ( !v17 )
    {
      v54 = 257;
      v17 = sub_B51D30(50, v10, v16, (__int64)&v51, 0, 0);
      if ( (unsigned __int8)sub_920620(v17) )
      {
        v39 = v15[12];
        v40 = *((_DWORD *)v15 + 26);
        if ( v39 )
          sub_B99FD0(v17, 3u, v39);
        sub_B45150(v17, v40);
      }
      (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v15[11] + 16LL))(
        v15[11],
        v17,
        v49,
        v15[7],
        v15[8]);
      v41 = *v15;
      v42 = *v15 + 16LL * *((unsigned int *)v15 + 2);
      while ( v42 != v41 )
      {
        v43 = *(_QWORD *)(v41 + 8);
        v44 = *(_DWORD *)v41;
        v41 += 16;
        sub_B99FD0(v17, v44, v43);
      }
    }
  }
  v18 = *(_DWORD *)(v7 + 4);
  v19 = 0;
  v51 = (__int64 *)v53;
  v20 = 32 * (1LL - (v18 & 0x7FFFFFF));
  v52 = 0x800000000LL;
  v21 = (__int64 *)v53;
  v22 = (__int64 *)(v7 + v20);
  v23 = -v20;
  v24 = v23 >> 5;
  if ( (unsigned __int64)v23 > 0x100 )
  {
    sub_C8D5F0((__int64)&v51, v53, v23 >> 5, 8u, v13, v14);
    v19 = v52;
    v21 = &v51[(unsigned int)v52];
  }
  if ( (__int64 *)v7 != v22 )
  {
    do
    {
      if ( v21 )
        *v21 = *v22;
      v22 += 4;
      ++v21;
    }
    while ( (__int64 *)v7 != v22 );
    v19 = v52;
  }
  LODWORD(v52) = v24 + v19;
  v25 = sub_BD5D20(v7);
  v26 = v51;
  v49[0] = v25;
  v50 = 261;
  v49[1] = v27;
  v47 = (unsigned int)v52;
  v46 = *(_QWORD *)(v7 + 72);
  v45 = v52 + 1;
  v28 = sub_BD2C40(88, (int)v52 + 1);
  if ( v28 )
  {
    v29 = *(_QWORD *)(v17 + 8);
    v30 = v45 & 0x7FFFFFF;
    if ( (unsigned int)*(unsigned __int8 *)(v29 + 8) - 17 > 1 )
    {
      v33 = &v26[v47];
      if ( v26 != v33 )
      {
        v34 = v26;
        while ( 1 )
        {
          v35 = *(_QWORD *)(*v34 + 8);
          v36 = *(unsigned __int8 *)(v35 + 8);
          if ( v36 == 17 )
          {
            v37 = 0;
            goto LABEL_27;
          }
          if ( v36 == 18 )
            break;
          if ( v33 == ++v34 )
            goto LABEL_16;
        }
        v37 = 1;
LABEL_27:
        BYTE4(v48) = v37;
        LODWORD(v48) = *(_DWORD *)(v35 + 32);
        v38 = sub_BCE1B0((__int64 *)v29, v48);
        v30 = v45 & 0x7FFFFFF;
        v29 = v38;
      }
    }
LABEL_16:
    sub_B44260((__int64)v28, v29, 34, v30, 0, 0);
    v28[9] = v46;
    v28[10] = sub_B4DC50(v46, (__int64)v26, v47);
    sub_B4D9A0((__int64)v28, v17, v26, v47, (__int64)v49);
  }
  v31 = sub_B4DE30(v7);
  v32 = v31;
  sub_B4DE00((__int64)v28, v31);
  if ( v51 != (__int64 *)v53 )
    _libc_free(v51, v32);
  return (unsigned __int8 *)v28;
}
