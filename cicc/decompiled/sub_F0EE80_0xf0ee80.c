// Function: sub_F0EE80
// Address: 0xf0ee80
//
unsigned __int8 *__fastcall sub_F0EE80(__int64 a1, unsigned __int8 *a2)
{
  _BYTE *v2; // rax
  unsigned __int8 *v3; // r13
  __int64 v5; // r15
  _BYTE *v6; // r14
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 *v11; // r15
  int v12; // r11d
  __int64 v13; // r12
  __int64 *v14; // r13
  int v15; // ebx
  unsigned int v16; // ebx
  __int64 v17; // r15
  unsigned __int8 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  int v26; // r14d
  __int64 v27; // rbx
  __int64 v28; // r13
  __int64 v29; // rdx
  unsigned int v30; // esi
  __int64 v31; // rdx
  int v32; // r8d
  __int64 v33; // r15
  __int64 v34; // rdx
  unsigned int v35; // esi
  __int64 v36; // [rsp+0h] [rbp-B0h]
  __int64 v37; // [rsp+8h] [rbp-A8h]
  __int64 v38; // [rsp+10h] [rbp-A0h]
  int v39; // [rsp+18h] [rbp-98h]
  int v40; // [rsp+18h] [rbp-98h]
  __int64 v41; // [rsp+18h] [rbp-98h]
  _BYTE v42[32]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v43; // [rsp+40h] [rbp-70h]
  const char *v44[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v45; // [rsp+70h] [rbp-40h]

  v2 = (_BYTE *)*((_QWORD *)a2 - 8);
  if ( *v2 != 69 )
    return 0;
  v5 = *((_QWORD *)v2 - 4);
  v37 = v5;
  if ( !v5 )
    return 0;
  v6 = (_BYTE *)*((_QWORD *)a2 - 4);
  if ( *v6 > 0x15u || *v6 == 5 || (unsigned __int8)sub_AD6CA0(*((_QWORD *)a2 - 4)) )
    return 0;
  v9 = *(_QWORD *)(v5 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
    v9 = **(_QWORD **)(v9 + 16);
  if ( !sub_BCAC40(v9, 1) )
    return 0;
  v36 = sub_AD62B0(*((_QWORD *)a2 + 1));
  v10 = sub_AD6530(*((_QWORD *)a2 + 1), 1);
  v11 = *(__int64 **)(a1 + 32);
  v12 = *a2;
  v38 = v10;
  v43 = 257;
  v39 = v12 - 29;
  v13 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, _BYTE *))(*(_QWORD *)v11[10] + 16LL))(
          v11[10],
          (unsigned int)(v12 - 29),
          v36,
          v6);
  if ( !v13 )
  {
    v45 = 257;
    v13 = sub_B504D0(v39, v36, (__int64)v6, (__int64)v44, 0, 0);
    if ( (unsigned __int8)sub_920620(v13) )
    {
      v31 = v11[12];
      v32 = *((_DWORD *)v11 + 26);
      if ( v31 )
      {
        v40 = *((_DWORD *)v11 + 26);
        sub_B99FD0(v13, 3u, v31);
        v32 = v40;
      }
      sub_B45150(v13, v32);
    }
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v11[11] + 16LL))(
      v11[11],
      v13,
      v42,
      v11[7],
      v11[8]);
    v41 = *v11 + 16LL * *((unsigned int *)v11 + 2);
    if ( *v11 != v41 )
    {
      v33 = *v11;
      do
      {
        v34 = *(_QWORD *)(v33 + 8);
        v35 = *(_DWORD *)v33;
        v33 += 16;
        sub_B99FD0(v13, v35, v34);
      }
      while ( v41 != v33 );
    }
  }
  v14 = *(__int64 **)(a1 + 32);
  v15 = *a2;
  v43 = 257;
  v16 = v15 - 29;
  v17 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, _BYTE *))(*(_QWORD *)v14[10] + 16LL))(
          v14[10],
          v16,
          v38,
          v6);
  if ( !v17 )
  {
    v45 = 257;
    v17 = sub_B504D0(v16, v38, (__int64)v6, (__int64)v44, 0, 0);
    if ( (unsigned __int8)sub_920620(v17) )
    {
      v25 = v14[12];
      v26 = *((_DWORD *)v14 + 26);
      if ( v25 )
        sub_B99FD0(v17, 3u, v25);
      sub_B45150(v17, v26);
    }
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v14[11] + 16LL))(
      v14[11],
      v17,
      v42,
      v14[7],
      v14[8]);
    v27 = *v14;
    v28 = *v14 + 16LL * *((unsigned int *)v14 + 2);
    while ( v28 != v27 )
    {
      v29 = *(_QWORD *)(v27 + 8);
      v30 = *(_DWORD *)v27;
      v27 += 16;
      sub_B99FD0(v17, v30, v29);
    }
  }
  v45 = 257;
  v18 = (unsigned __int8 *)sub_BD2C40(72, 3u);
  v3 = v18;
  if ( v18 )
  {
    sub_B44260((__int64)v18, *(_QWORD *)(v13 + 8), 57, 3u, 0, 0);
    if ( *((_QWORD *)v3 - 12) )
    {
      v19 = *((_QWORD *)v3 - 11);
      **((_QWORD **)v3 - 10) = v19;
      if ( v19 )
        *(_QWORD *)(v19 + 16) = *((_QWORD *)v3 - 10);
    }
    *((_QWORD *)v3 - 12) = v37;
    v20 = *(_QWORD *)(v37 + 16);
    *((_QWORD *)v3 - 11) = v20;
    if ( v20 )
      *(_QWORD *)(v20 + 16) = v3 - 88;
    *((_QWORD *)v3 - 10) = v37 + 16;
    *(_QWORD *)(v37 + 16) = v3 - 96;
    if ( *((_QWORD *)v3 - 8) )
    {
      v21 = *((_QWORD *)v3 - 7);
      **((_QWORD **)v3 - 6) = v21;
      if ( v21 )
        *(_QWORD *)(v21 + 16) = *((_QWORD *)v3 - 6);
    }
    *((_QWORD *)v3 - 8) = v13;
    v22 = *(_QWORD *)(v13 + 16);
    *((_QWORD *)v3 - 7) = v22;
    if ( v22 )
      *(_QWORD *)(v22 + 16) = v3 - 56;
    *((_QWORD *)v3 - 6) = v13 + 16;
    *(_QWORD *)(v13 + 16) = v3 - 64;
    if ( *((_QWORD *)v3 - 4) )
    {
      v23 = *((_QWORD *)v3 - 3);
      **((_QWORD **)v3 - 2) = v23;
      if ( v23 )
        *(_QWORD *)(v23 + 16) = *((_QWORD *)v3 - 2);
    }
    *((_QWORD *)v3 - 4) = v17;
    if ( v17 )
    {
      v24 = *(_QWORD *)(v17 + 16);
      *((_QWORD *)v3 - 3) = v24;
      if ( v24 )
        *(_QWORD *)(v24 + 16) = v3 - 24;
      *((_QWORD *)v3 - 2) = v17 + 16;
      *(_QWORD *)(v17 + 16) = v3 - 32;
    }
    sub_BD6B50(v3, v44);
  }
  return v3;
}
