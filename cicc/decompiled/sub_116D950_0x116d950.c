// Function: sub_116D950
// Address: 0x116d950
//
__int64 __fastcall sub_116D950(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v3; // r14
  unsigned __int8 *v5; // r12
  __int64 v6; // r13
  unsigned __int8 v7; // al
  unsigned __int8 v8; // al
  unsigned __int8 *v9; // rdx
  unsigned __int8 *v10; // rdi
  __int64 v11; // r12
  _BYTE *v13; // rdx
  unsigned __int8 *v14; // rax
  int v15; // edi
  __int64 v16; // rsi
  unsigned __int8 *v17; // r15
  __int64 v18; // rax
  __int16 v19; // dx
  __int64 v20; // r8
  char v21; // al
  char v22; // dl
  __int16 v23; // cx
  __int64 v24; // rdi
  unsigned int v25; // r14d
  unsigned __int8 v26; // r14
  __int64 v27; // r14
  __int64 v28; // r11
  __int64 v29; // r13
  __int64 v30; // rbx
  __int64 v31; // r13
  __int64 v32; // rdx
  unsigned int v33; // esi
  unsigned __int8 *v34; // rax
  unsigned __int8 **v35; // rax
  __int64 v36; // r13
  int v37; // ecx
  unsigned __int8 v38; // al
  unsigned __int8 v39; // al
  __int64 v40; // rdx
  int v41; // r13d
  __int64 v42; // r13
  __int64 v43; // rbx
  __int64 v44; // rdx
  unsigned int v45; // esi
  __int64 v46; // rdx
  int v47; // eax
  char v48; // cl
  int v49; // eax
  __int64 v50; // [rsp+8h] [rbp-C8h]
  unsigned int v51; // [rsp+8h] [rbp-C8h]
  __int64 v52; // [rsp+18h] [rbp-B8h] BYREF
  __int64 v53; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v54; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v55; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v56; // [rsp+38h] [rbp-98h]
  _BYTE v57[32]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v58; // [rsp+60h] [rbp-70h]
  _BYTE v59[32]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v60; // [rsp+90h] [rbp-40h]

  if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != 2 )
    return 0;
  v2 = *(__int64 **)(a1 - 8);
  v3 = *(_QWORD *)(a1 + 40);
  v5 = (unsigned __int8 *)v2[4];
  v6 = *v2;
  v7 = *v5;
  if ( *v5 <= 0x1Cu )
  {
LABEL_3:
    v8 = *(_BYTE *)v6;
    if ( *(_BYTE *)v6 > 0x1Cu )
    {
      if ( (unsigned int)v8 - 42 <= 0x11 )
      {
        v9 = *(unsigned __int8 **)(v6 - 64);
        v10 = *(unsigned __int8 **)(v6 - 32);
        if ( v9 && v5 == v9 && (unsigned __int8)(*v10 - 42) <= 0x11u )
          goto LABEL_44;
        if ( v5 == v10 && v10 && (unsigned __int8)(*v9 - 42) <= 0x11u )
        {
          v10 = *(unsigned __int8 **)(v6 - 64);
LABEL_44:
          v34 = (unsigned __int8 *)v6;
          v6 = (__int64)v5;
          v5 = v34;
          goto LABEL_21;
        }
      }
      if ( v8 == 63 && (*(_DWORD *)(v6 + 4) & 0x7FFFFFF) == 2 )
      {
        if ( (*(_BYTE *)(v6 + 7) & 0x40) != 0 )
        {
          v35 = *(unsigned __int8 ***)(v6 - 8);
          if ( v5 != *v35 )
            return 0;
        }
        else
        {
          if ( v5 != *(unsigned __int8 **)(v6 - 64) )
            return 0;
          v35 = (unsigned __int8 **)(v6 - 64);
        }
        v10 = v35[4];
        if ( (unsigned __int8)(*v10 - 42) <= 0x11u )
          goto LABEL_44;
      }
    }
    return 0;
  }
  if ( (unsigned int)v7 - 42 > 0x11 )
    goto LABEL_17;
  v13 = (_BYTE *)*((_QWORD *)v5 - 8);
  v10 = (unsigned __int8 *)*((_QWORD *)v5 - 4);
  if ( (_BYTE *)v6 == v13 && (unsigned __int8)(*v10 - 42) <= 0x11u )
    goto LABEL_21;
  if ( (unsigned __int8 *)v6 != v10 || (unsigned __int8)(*v13 - 42) > 0x11u )
  {
LABEL_17:
    if ( v7 != 63 || (*((_DWORD *)v5 + 1) & 0x7FFFFFF) != 2 )
      goto LABEL_3;
    if ( (v5[7] & 0x40) != 0 )
    {
      v14 = (unsigned __int8 *)*((_QWORD *)v5 - 1);
      if ( v6 != *(_QWORD *)v14 )
        goto LABEL_3;
    }
    else
    {
      if ( v6 != *((_QWORD *)v5 - 8) )
        goto LABEL_3;
      v14 = v5 - 64;
    }
    v10 = (unsigned __int8 *)*((_QWORD *)v14 + 4);
    if ( (unsigned __int8)(*v10 - 42) <= 0x11u )
      goto LABEL_21;
    goto LABEL_3;
  }
  v10 = (unsigned __int8 *)*((_QWORD *)v5 - 8);
LABEL_21:
  if ( !sub_9913D0((__int64)v10, &v52, &v53, &v54) || v3 != *(_QWORD *)(v52 + 40) )
    return 0;
  v15 = *v5;
  v16 = *(_QWORD *)(v53 + 8);
  if ( (unsigned int)(v15 - 42) > 0x11 )
  {
    v17 = 0;
    v18 = sub_AD6530(*(_QWORD *)(v53 + 8), v16);
  }
  else
  {
    v17 = v5;
    v18 = (__int64)sub_AD93D0(v15 - 29, v16, 0, 0);
  }
  if ( v53 != v18 )
    return 0;
  v20 = sub_AA5190(v3);
  if ( v20 )
  {
    v21 = v19;
    v22 = HIBYTE(v19);
  }
  else
  {
    v22 = 0;
    v21 = 0;
  }
  LOBYTE(v23) = v21;
  HIBYTE(v23) = v22;
  sub_A88F30(a2, v3, v20, v23);
  if ( v17 )
  {
    v24 = *(_QWORD *)(a2 + 80);
    v58 = 257;
    v50 = v52;
    v25 = *v17 - 29;
    v11 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v24 + 16LL))(v24, v25, v52, v6);
    if ( !v11 )
    {
      v60 = 257;
      v11 = sub_B504D0(v25, v50, v6, (__int64)v59, 0, 0);
      if ( *(_BYTE *)v11 > 0x1Cu )
      {
        switch ( *(_BYTE *)v11 )
        {
          case ')':
          case '+':
          case '-':
          case '/':
          case '2':
          case '5':
          case 'J':
          case 'K':
          case 'S':
            goto LABEL_69;
          case 'T':
          case 'U':
          case 'V':
            v36 = *(_QWORD *)(v11 + 8);
            v37 = *(unsigned __int8 *)(v36 + 8);
            v38 = *(_BYTE *)(v36 + 8);
            if ( (unsigned int)(v37 - 17) <= 1 )
              v38 = *(_BYTE *)(**(_QWORD **)(v36 + 16) + 8LL);
            if ( v38 <= 3u || v38 == 5 || (v38 & 0xFD) == 4 )
              goto LABEL_69;
            if ( (_BYTE)v37 == 15 )
            {
              if ( (*(_BYTE *)(v36 + 9) & 4) == 0 || !sub_BCB420(*(_QWORD *)(v11 + 8)) )
                break;
              v36 = **(_QWORD **)(v36 + 16);
            }
            else if ( (_BYTE)v37 == 16 )
            {
              do
                v36 = *(_QWORD *)(v36 + 24);
              while ( *(_BYTE *)(v36 + 8) == 16 );
            }
            if ( (unsigned int)*(unsigned __int8 *)(v36 + 8) - 17 <= 1 )
              v36 = **(_QWORD **)(v36 + 16);
            v39 = *(_BYTE *)(v36 + 8);
            if ( v39 <= 3u || v39 == 5 || (v39 & 0xFD) == 4 )
            {
LABEL_69:
              v40 = *(_QWORD *)(a2 + 96);
              v41 = *(_DWORD *)(a2 + 104);
              if ( v40 )
                sub_B99FD0(v11, 3u, v40);
              sub_B45150(v11, v41);
            }
            break;
          default:
            break;
        }
      }
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v11,
        v57,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v42 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v42 )
      {
        v43 = *(_QWORD *)a2;
        do
        {
          v44 = *(_QWORD *)(v43 + 8);
          v45 = *(_DWORD *)v43;
          v43 += 16;
          sub_B99FD0(v11, v45, v44);
        }
        while ( v42 != v43 );
      }
    }
    sub_B45260((unsigned __int8 *)v11, (__int64)v17, 1);
    return v11;
  }
  v26 = v5[1];
  v58 = 257;
  v51 = v26 >> 1;
  v55 = v52;
  v27 = sub_BB5290((__int64)v5);
  v11 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64 *, __int64, _QWORD))(**(_QWORD **)(a2 + 80) + 64LL))(
          *(_QWORD *)(a2 + 80),
          v27,
          v6,
          &v55,
          1,
          v51);
  if ( v11 )
    return v11;
  v60 = 257;
  v11 = (__int64)sub_BD2C40(88, 2u);
  if ( v11 )
  {
    v28 = *(_QWORD *)(v6 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v28 + 8) - 17 <= 1 )
    {
LABEL_37:
      sub_B44260(v11, v28, 34, 2u, 0, 0);
      *(_QWORD *)(v11 + 72) = v27;
      *(_QWORD *)(v11 + 80) = sub_B4DC50(v27, (__int64)&v55, 1);
      sub_B4D9A0(v11, v6, &v55, 1, (__int64)v59);
      goto LABEL_38;
    }
    v46 = *(_QWORD *)(v55 + 8);
    v47 = *(unsigned __int8 *)(v46 + 8);
    if ( v47 == 17 )
    {
      v48 = 0;
    }
    else
    {
      v48 = 1;
      if ( v47 != 18 )
        goto LABEL_37;
    }
    v49 = *(_DWORD *)(v46 + 32);
    BYTE4(v56) = v48;
    LODWORD(v56) = v49;
    v28 = sub_BCE1B0((__int64 *)v28, v56);
    goto LABEL_37;
  }
LABEL_38:
  sub_B4DDE0(v11, v51);
  (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v11,
    v57,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  v29 = 16LL * *(unsigned int *)(a2 + 8);
  v30 = *(_QWORD *)a2;
  v31 = v30 + v29;
  while ( v31 != v30 )
  {
    v32 = *(_QWORD *)(v30 + 8);
    v33 = *(_DWORD *)v30;
    v30 += 16;
    sub_B99FD0(v11, v33, v32);
  }
  return v11;
}
