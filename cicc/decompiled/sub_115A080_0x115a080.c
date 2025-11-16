// Function: sub_115A080
// Address: 0x115a080
//
unsigned __int8 *__fastcall sub_115A080(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v6; // r14
  int v7; // r13d
  unsigned __int64 v8; // rbx
  unsigned int v9; // r13d
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r10
  __int64 v19; // rax
  __int64 v20; // r11
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 *v23; // rbx
  int v24; // r14d
  __int64 v25; // r9
  __int64 v26; // r13
  const char *v27; // rax
  _QWORD *v28; // rdx
  __int64 v29; // rax
  char v30; // al
  __int64 v31; // r9
  __int64 v32; // rdx
  __int64 v33; // r15
  __int64 v34; // r13
  __int64 v35; // rdx
  unsigned int v36; // esi
  __int64 v37; // [rsp+8h] [rbp-B8h]
  __int64 v38; // [rsp+8h] [rbp-B8h]
  __int64 v39; // [rsp+8h] [rbp-B8h]
  __int64 v40; // [rsp+8h] [rbp-B8h]
  __int64 v41; // [rsp+8h] [rbp-B8h]
  __int64 v42; // [rsp+8h] [rbp-B8h]
  __int64 v43; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v44; // [rsp+18h] [rbp-A8h]
  __int64 v45; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v46; // [rsp+28h] [rbp-98h] BYREF
  _QWORD *v47[4]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v48; // [rsp+50h] [rbp-70h]
  _QWORD *v49[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v50; // [rsp+80h] [rbp-40h]

  v6 = *((_QWORD *)a2 - 8);
  v7 = *a2;
  v8 = *((_QWORD *)a2 - 4);
  v47[0] = &v45;
  v9 = v7 - 29;
  if ( (unsigned __int8)sub_995E90(v47, v6, a3, a4, a5) )
  {
    v49[0] = &v46;
    if ( (unsigned __int8)sub_995E90(v49, v8, v10, v11, v12) )
    {
      v14 = v46;
      v15 = v45;
      v50 = 257;
LABEL_7:
      v44 = (unsigned __int8 *)sub_B504D0(v9, v15, v14, (__int64)v49, 0, 0);
      sub_B45260(v44, (__int64)a2, 1);
      return v44;
    }
  }
  if ( v8 == v6 )
  {
    if ( *(_BYTE *)v8 != 85 )
      return 0;
    v16 = *(_QWORD *)(v8 - 32);
    if ( v16 )
    {
      if ( !*(_BYTE *)v16 && *(_QWORD *)(v16 + 24) == *(_QWORD *)(v8 + 80) && *(_DWORD *)(v16 + 36) == 170 )
      {
        v15 = *(_QWORD *)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
        if ( v15 )
        {
          v45 = *(_QWORD *)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
          v14 = v15;
          v50 = 257;
          goto LABEL_7;
        }
      }
    }
  }
  else if ( *(_BYTE *)v6 != 85 )
  {
    return 0;
  }
  v17 = *(_QWORD *)(v6 - 32);
  if ( !v17 )
    return 0;
  if ( *(_BYTE *)v17 )
    return 0;
  if ( *(_QWORD *)(v17 + 24) != *(_QWORD *)(v6 + 80) )
    return 0;
  if ( *(_DWORD *)(v17 + 36) != 170 )
    return 0;
  v18 = *(_QWORD *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
  if ( !v18 )
    return 0;
  v45 = *(_QWORD *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
  if ( *(_BYTE *)v8 != 85 )
    return 0;
  v19 = *(_QWORD *)(v8 - 32);
  if ( !v19 )
    return 0;
  if ( *(_BYTE *)v19 )
    return 0;
  if ( *(_QWORD *)(v19 + 24) != *(_QWORD *)(v8 + 80) )
    return 0;
  if ( *(_DWORD *)(v19 + 36) != 170 )
    return 0;
  v20 = *(_QWORD *)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
  if ( !v20 )
    return 0;
  v21 = *(_QWORD *)(v6 + 16);
  v46 = *(_QWORD *)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
  if ( !v21 || *(_QWORD *)(v21 + 8) )
  {
    v22 = *(_QWORD *)(v8 + 16);
    if ( !v22 || *(_QWORD *)(v22 + 8) )
      return 0;
  }
  v37 = v20;
  v43 = v18;
  v23 = *(__int64 **)(a1 + 32);
  v48 = 257;
  v24 = sub_B45210((__int64)a2);
  v25 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v23[10] + 16LL))(
          v23[10],
          v9,
          v43,
          v37);
  if ( !v25 )
  {
    v50 = 257;
    v39 = sub_B504D0(v9, v43, v37, (__int64)v49, 0, 0);
    v30 = sub_920620(v39);
    v31 = v39;
    if ( v30 )
    {
      v32 = v23[12];
      if ( v32 )
      {
        sub_B99FD0(v39, 3u, v32);
        v31 = v39;
      }
      v40 = v31;
      sub_B45150(v31, v24);
      v31 = v40;
    }
    v41 = v31;
    (*(void (__fastcall **)(__int64, __int64, _QWORD **, __int64, __int64))(*(_QWORD *)v23[11] + 16LL))(
      v23[11],
      v31,
      v47,
      v23[7],
      v23[8]);
    v33 = *v23;
    v25 = v41;
    v34 = *v23 + 16LL * *((unsigned int *)v23 + 2);
    if ( *v23 != v34 )
    {
      do
      {
        v35 = *(_QWORD *)(v33 + 8);
        v36 = *(_DWORD *)v33;
        v33 += 16;
        v42 = v25;
        sub_B99FD0(v25, v36, v35);
        v25 = v42;
      }
      while ( v34 != v33 );
    }
  }
  v38 = v25;
  v26 = *(_QWORD *)(a1 + 32);
  v27 = sub_BD5D20((__int64)a2);
  v50 = 261;
  v49[1] = v28;
  v49[0] = v27;
  LODWORD(v47[0]) = sub_B45210((__int64)a2);
  BYTE4(v47[0]) = 1;
  v29 = sub_B33BC0(v26, 0xAAu, v38, (__int64)v47[0], (__int64)v49);
  return sub_F162A0(a1, (__int64)a2, v29);
}
