// Function: sub_1197150
// Address: 0x1197150
//
_QWORD *__fastcall sub_1197150(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rdx
  __int64 v5; // rdi
  __int64 *v6; // r14
  _BYTE *v7; // rdi
  _QWORD *result; // rax
  _BYTE *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rcx
  __int64 v14; // rax
  _QWORD *v15; // rcx
  __int64 v16; // rcx
  __int64 *v17; // r10
  __int64 v18; // rbx
  __int64 *v19; // r10
  __int64 v20; // rax
  __int64 *v21; // r11
  __int64 v22; // r15
  __int64 v23; // rsi
  unsigned int v24; // eax
  __int64 v25; // rax
  _QWORD *v26; // rax
  __int64 *v27; // r10
  _QWORD **v28; // rdx
  int v29; // ecx
  __int64 *v30; // rax
  __int64 v31; // rax
  __int64 v32; // r15
  __int64 v33; // rdx
  unsigned int v34; // esi
  __int64 v35; // rax
  _QWORD *v36; // rax
  __int64 *v37; // r11
  __int64 v38; // rbx
  __int64 v39; // rdx
  unsigned int v40; // esi
  __int64 v41; // [rsp+8h] [rbp-B8h]
  unsigned int v42; // [rsp+10h] [rbp-B0h]
  __int64 *v43; // [rsp+10h] [rbp-B0h]
  __int64 *v44; // [rsp+10h] [rbp-B0h]
  __int64 *v45; // [rsp+10h] [rbp-B0h]
  __int64 v46; // [rsp+10h] [rbp-B0h]
  __int64 v47; // [rsp+10h] [rbp-B0h]
  __int64 *v48; // [rsp+10h] [rbp-B0h]
  __int64 v49; // [rsp+10h] [rbp-B0h]
  __int64 *v50; // [rsp+18h] [rbp-A8h]
  __int64 v51; // [rsp+18h] [rbp-A8h]
  _QWORD *v52; // [rsp+18h] [rbp-A8h]
  __int64 v53; // [rsp+18h] [rbp-A8h]
  __int64 v54; // [rsp+18h] [rbp-A8h]
  __int64 v55; // [rsp+28h] [rbp-98h]
  _QWORD v56[4]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v57; // [rsp+50h] [rbp-70h]
  _BYTE v58[32]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v59; // [rsp+80h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)sub_BCB060(v3) <= 2 )
    return 0;
  v5 = *(_QWORD *)(a2 - 32);
  v6 = *(__int64 **)(a2 - 64);
  if ( *(_BYTE *)v5 == 17 )
  {
    v7 = (_BYTE *)(v5 + 24);
    goto LABEL_4;
  }
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 8LL) - 17 > 1 )
    return 0;
  if ( *(_BYTE *)v5 > 0x15u )
    return 0;
  v9 = sub_AD7630(v5, 0, v4);
  if ( !v9 || *v9 != 17 )
    return 0;
  v7 = v9 + 24;
LABEL_4:
  if ( *(_BYTE *)v6 != 42 )
    return 0;
  v10 = *(v6 - 8);
  v11 = *(_QWORD *)(v10 + 16);
  if ( !v11 )
    return 0;
  if ( *(_QWORD *)(v11 + 8) )
    return 0;
  if ( *(_BYTE *)v10 != 68 )
    return 0;
  v12 = *(_QWORD *)(v10 - 32);
  if ( !v12 )
    return 0;
  v13 = *(v6 - 4);
  v14 = *(_QWORD *)(v13 + 16);
  if ( !v14 )
    return 0;
  if ( *(_QWORD *)(v14 + 8) )
    return 0;
  if ( *(_BYTE *)v13 != 68 )
    return 0;
  v41 = *(_QWORD *)(v13 - 32);
  if ( !v41 )
    return 0;
  v15 = *(_QWORD **)v7;
  if ( *((_DWORD *)v7 + 2) > 0x40u )
    v15 = **(_QWORD ***)v7;
  v42 = (unsigned int)v15;
  if ( (_DWORD)v15 == 1
    || (unsigned int)sub_BCB060(*(_QWORD *)(v12 + 8)) != (_DWORD)v15
    || (unsigned int)sub_BCB060(*(_QWORD *)(v41 + 8)) != v42 )
  {
    return 0;
  }
  v16 = v6[2];
  if ( v16 && *(_QWORD *)(v16 + 8) )
  {
    do
    {
      v25 = *(_QWORD *)(v16 + 24);
      if ( a2 != v25 )
      {
        if ( *(_BYTE *)v25 != 67 )
          return 0;
        v53 = v16;
        v24 = sub_BCB060(*(_QWORD *)(v25 + 8));
        v16 = v53;
        if ( v24 > v42 )
          return 0;
      }
      v16 = *(_QWORD *)(v16 + 8);
    }
    while ( v16 );
  }
  sub_D5F1F0(*(_QWORD *)(a1 + 32), (__int64)v6);
  v17 = *(__int64 **)(a1 + 32);
  v56[0] = "add.narrowed";
  v57 = 259;
  v50 = v17;
  v18 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v17[10] + 32LL))(
          v17[10],
          13,
          v12,
          v41,
          0,
          0);
  if ( !v18 )
  {
    v59 = 257;
    v18 = sub_B504D0(13, v12, v41, (__int64)v58, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v50[11] + 16LL))(
      v50[11],
      v18,
      v56,
      v50[7],
      v50[8]);
    v35 = *v50;
    v47 = *v50 + 16LL * *((unsigned int *)v50 + 2);
    if ( *v50 != v47 )
    {
      do
      {
        v54 = v35;
        sub_B99FD0(v18, *(_DWORD *)v35, *(_QWORD *)(v35 + 8));
        v35 = v54 + 16;
      }
      while ( v47 != v54 + 16 );
    }
  }
  v19 = *(__int64 **)(a1 + 32);
  v57 = 259;
  v56[0] = "add.narrowed.overflow";
  v43 = v19;
  v51 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v19[10] + 56LL))(
          v19[10],
          36,
          v18,
          v12);
  if ( !v51 )
  {
    v59 = 257;
    v26 = sub_BD2C40(72, unk_3F10FD0);
    v27 = v43;
    v51 = (__int64)v26;
    if ( v26 )
    {
      v28 = *(_QWORD ***)(v18 + 8);
      v29 = *((unsigned __int8 *)v28 + 8);
      if ( (unsigned int)(v29 - 17) > 1 )
      {
        v31 = sub_BCB2A0(*v28);
      }
      else
      {
        BYTE4(v55) = (_BYTE)v29 == 18;
        LODWORD(v55) = *((_DWORD *)v28 + 8);
        v30 = (__int64 *)sub_BCB2A0(*v28);
        v31 = sub_BCE1B0(v30, v55);
      }
      sub_B523C0(v51, v31, 53, 36, v18, v12, (__int64)v58, 0, 0, 0);
      v27 = v43;
    }
    v45 = v27;
    (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v27[11] + 16LL))(
      v27[11],
      v51,
      v56,
      v27[7],
      v27[8]);
    v32 = *v45;
    v46 = *v45 + 16LL * *((unsigned int *)v45 + 2);
    while ( v46 != v32 )
    {
      v33 = *(_QWORD *)(v32 + 8);
      v34 = *(_DWORD *)v32;
      v32 += 16;
      sub_B99FD0(v51, v34, v33);
    }
  }
  v20 = v6[2];
  if ( !v20 || *(_QWORD *)(v20 + 8) )
  {
    v21 = *(__int64 **)(a1 + 32);
    v57 = 257;
    if ( v3 == *(_QWORD *)(v18 + 8) )
    {
      v22 = v18;
    }
    else
    {
      v44 = v21;
      v22 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v21[10] + 120LL))(
              v21[10],
              39,
              v18,
              v3);
      if ( !v22 )
      {
        v59 = 257;
        v36 = sub_BD2C40(72, unk_3F10A14);
        v37 = v44;
        v22 = (__int64)v36;
        if ( v36 )
        {
          sub_B515B0((__int64)v36, v18, v3, (__int64)v58, 0, 0);
          v37 = v44;
        }
        v48 = v37;
        (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v37[11] + 16LL))(
          v37[11],
          v22,
          v56,
          v37[7],
          v37[8]);
        v38 = *v48;
        v49 = *v48 + 16LL * *((unsigned int *)v48 + 2);
        while ( v49 != v38 )
        {
          v39 = *(_QWORD *)(v38 + 8);
          v40 = *(_DWORD *)v38;
          v38 += 16;
          sub_B99FD0(v22, v40, v39);
        }
      }
    }
    sub_F162A0(a1, (__int64)v6, v22);
    sub_F207A0(a1, v6);
  }
  v59 = 257;
  result = sub_BD2C40(72, unk_3F10A14);
  if ( result )
  {
    v23 = v51;
    v52 = result;
    sub_B515B0((__int64)result, v23, v3, (__int64)v58, 0, 0);
    return v52;
  }
  return result;
}
