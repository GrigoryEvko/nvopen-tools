// Function: sub_10E5480
// Address: 0x10e5480
//
unsigned __int8 *__fastcall sub_10E5480(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  unsigned __int8 *v9; // r8
  __int64 v10; // rdx
  unsigned int v11; // r14d
  __int64 v12; // rdi
  bool v13; // r14
  __int64 v14; // rax
  int v16; // edx
  __int64 v17; // rcx
  __int64 v18; // rdi
  bool v19; // al
  _BYTE *v20; // rax
  __int64 v21; // rdx
  unsigned int v22; // r14d
  __int64 *v23; // r14
  _QWORD *v24; // r15
  unsigned int **v25; // r14
  __int64 v26; // rax
  __int64 v27; // rax
  _QWORD **v28; // rdx
  int v29; // ecx
  __int64 *v30; // rax
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // r14
  __int64 v34; // rdx
  unsigned int v35; // esi
  __int64 v36; // rdx
  _BYTE *v37; // rax
  __int64 v38; // r15
  _BYTE *v39; // rax
  unsigned __int8 *v40; // rdx
  unsigned int v41; // r14d
  int v42; // eax
  unsigned int v43; // r15d
  __int64 v44; // rax
  unsigned int v45; // r14d
  int v46; // eax
  unsigned __int8 *v47; // [rsp+0h] [rbp-B0h]
  __int64 v48; // [rsp+8h] [rbp-A8h]
  __int64 v49; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v50; // [rsp+8h] [rbp-A8h]
  int v51; // [rsp+8h] [rbp-A8h]
  __int64 v52; // [rsp+18h] [rbp-98h]
  _BYTE v53[32]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v54; // [rsp+40h] [rbp-70h]
  _BYTE v55[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v56; // [rsp+70h] [rbp-40h]

  v6 = *a2;
  if ( (unsigned __int8)v6 <= 0x1Cu )
  {
    if ( (_BYTE)v6 != 5 )
      return 0;
    v16 = *((unsigned __int16 *)a2 + 1);
    if ( (*((_WORD *)a2 + 1) & 0xFFF7) != 0x11 && (v16 & 0xFFFD) != 0xD || (_WORD)v16 != 25 || (a2[1] & 2) == 0 )
      goto LABEL_21;
  }
  else
  {
    if ( (unsigned __int8)v6 > 0x36u )
      return 0;
    v7 = 0x40540000000000LL;
    if ( !_bittest64(&v7, v6) || (_BYTE)v6 != 54 || (a2[1] & 2) == 0 )
      goto LABEL_28;
  }
  v8 = *((_QWORD *)a2 - 8);
  if ( v8 )
  {
    v9 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
    v10 = *v9;
    if ( (_BYTE)v10 == 17 )
    {
      v11 = *((_DWORD *)v9 + 8);
      v12 = (__int64)(v9 + 24);
      goto LABEL_9;
    }
    if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v9 + 1) + 8LL) - 17 <= 1 && (unsigned __int8)v10 <= 0x15u )
    {
      v20 = sub_AD7630(*((_QWORD *)a2 - 4), 0, v10);
      if ( v20 && *v20 == 17 )
      {
        v11 = *((_DWORD *)v20 + 8);
        v12 = (__int64)(v20 + 24);
        goto LABEL_9;
      }
      v6 = *a2;
    }
  }
  if ( (unsigned __int8)v6 > 0x1Cu )
  {
    if ( (unsigned __int8)v6 > 0x36u )
      return 0;
LABEL_28:
    v17 = 0x40540000000000LL;
    v16 = (unsigned __int8)v6 - 29;
    if ( !_bittest64(&v17, v6) )
      return 0;
    goto LABEL_29;
  }
  if ( (_BYTE)v6 != 5 )
    return 0;
  v16 = *((unsigned __int16 *)a2 + 1);
LABEL_21:
  if ( (v16 & 0xFFF7) != 0x11 && (v16 & 0xFFFD) != 0xD )
    return 0;
LABEL_29:
  if ( v16 != 17 )
    return 0;
  if ( (a2[1] & 2) == 0 )
    return 0;
  v8 = *((_QWORD *)a2 - 8);
  if ( !v8 )
    return 0;
  v18 = *((_QWORD *)a2 - 4);
  if ( *(_BYTE *)v18 == 17 )
  {
    v12 = v18 + 24;
  }
  else
  {
    v36 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v18 + 8) + 8LL) - 17;
    if ( (unsigned int)v36 > 1 )
      return 0;
    if ( *(_BYTE *)v18 > 0x15u )
      return 0;
    v37 = sub_AD7630(v18, 0, v36);
    if ( !v37 || *v37 != 17 )
      return 0;
    v12 = (__int64)(v37 + 24);
  }
  v11 = *(_DWORD *)(v12 + 8);
  if ( v11 <= 0x40 )
    v19 = *(_QWORD *)v12 == 1;
  else
    v19 = v11 - 1 == (unsigned int)sub_C444A0(v12);
  if ( v19 )
    return 0;
LABEL_9:
  if ( v11 <= 0x40 )
    v13 = *(_QWORD *)v12 == 0;
  else
    v13 = v11 == (unsigned int)sub_C444A0(v12);
  if ( v13 )
    return 0;
  v14 = *(_QWORD *)(a3 + 16);
  if ( !v14 || *(_QWORD *)(v14 + 8) || *(_BYTE *)a3 != 42 || *(_QWORD *)(a3 - 64) != v8 )
    return 0;
  v21 = *(_QWORD *)(a3 - 32);
  if ( *(_BYTE *)v21 != 17 )
  {
    v38 = *(_QWORD *)(v21 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v38 + 8) - 17 <= 1 && *(_BYTE *)v21 <= 0x15u )
    {
      v50 = (unsigned __int8 *)v21;
      v39 = sub_AD7630(v21, 0, v21);
      v40 = v50;
      if ( v39 && *v39 == 17 )
      {
        v41 = *((_DWORD *)v39 + 8);
        if ( v41 <= 0x40 )
        {
          if ( *((_QWORD *)v39 + 3) == 1 )
            goto LABEL_46;
        }
        else if ( (unsigned int)sub_C444A0((__int64)(v39 + 24)) == v41 - 1 )
        {
          goto LABEL_46;
        }
      }
      else if ( *(_BYTE *)(v38 + 8) == 17 )
      {
        v42 = *(_DWORD *)(v38 + 32);
        v43 = 0;
        v51 = v42;
        while ( v51 != v43 )
        {
          v47 = v40;
          v44 = sub_AD69F0(v40, v43);
          if ( !v44 )
            return 0;
          v40 = v47;
          if ( *(_BYTE *)v44 != 13 )
          {
            if ( *(_BYTE *)v44 != 17 )
              return 0;
            v45 = *(_DWORD *)(v44 + 32);
            if ( v45 <= 0x40 )
            {
              v13 = *(_QWORD *)(v44 + 24) == 1;
            }
            else
            {
              v46 = sub_C444A0(v44 + 24);
              v40 = v47;
              v13 = v45 - 1 == v46;
            }
            if ( !v13 )
              return 0;
          }
          ++v43;
        }
        if ( v13 )
          goto LABEL_46;
      }
    }
    return 0;
  }
  v22 = *(_DWORD *)(v21 + 32);
  if ( v22 <= 0x40 )
  {
    if ( *(_QWORD *)(v21 + 24) == 1 )
      goto LABEL_46;
    return 0;
  }
  if ( (unsigned int)sub_C444A0(v21 + 24) != v22 - 1 )
    return 0;
LABEL_46:
  v23 = *(__int64 **)(*(_QWORD *)a1 + 32LL);
  v54 = 257;
  v48 = sub_AD64C0(*(_QWORD *)(v8 + 8), 0, 0);
  v24 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v23[10] + 56LL))(
                    v23[10],
                    32,
                    v8,
                    v48);
  if ( !v24 )
  {
    v56 = 257;
    v24 = sub_BD2C40(72, unk_3F10FD0);
    if ( v24 )
    {
      v28 = *(_QWORD ***)(v8 + 8);
      v29 = *((unsigned __int8 *)v28 + 8);
      if ( (unsigned int)(v29 - 17) > 1 )
      {
        v31 = sub_BCB2A0(*v28);
      }
      else
      {
        BYTE4(v52) = (_BYTE)v29 == 18;
        LODWORD(v52) = *((_DWORD *)v28 + 8);
        v30 = (__int64 *)sub_BCB2A0(*v28);
        v31 = sub_BCE1B0(v30, v52);
      }
      sub_B523C0((__int64)v24, v31, 53, 32, v8, v48, (__int64)v55, 0, 0, 0);
    }
    (*(void (__fastcall **)(__int64, _QWORD *, _BYTE *, __int64, __int64))(*(_QWORD *)v23[11] + 16LL))(
      v23[11],
      v24,
      v53,
      v23[7],
      v23[8]);
    v32 = *v23 + 16LL * *((unsigned int *)v23 + 2);
    v33 = *v23;
    v49 = v32;
    while ( v49 != v33 )
    {
      v34 = *(_QWORD *)(v33 + 8);
      v35 = *(_DWORD *)v33;
      v33 += 16;
      sub_B99FD0((__int64)v24, v35, v34);
    }
  }
  v25 = *(unsigned int ***)(*(_QWORD *)a1 + 32LL);
  v56 = 257;
  v26 = sub_AD64C0(*(_QWORD *)(v8 + 8), 1, 0);
  v27 = sub_B36550(v25, (__int64)v24, v26, (__int64)a2, (__int64)v55, 0);
  return sub_F162A0(*(_QWORD *)a1, **(_QWORD **)(a1 + 8), v27);
}
