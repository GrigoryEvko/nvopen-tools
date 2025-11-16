// Function: sub_10BF450
// Address: 0x10bf450
//
_QWORD *__fastcall sub_10BF450(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // r15
  __int64 v3; // rax
  _QWORD *v4; // r12
  char v6; // al
  _BYTE *v8; // rdi
  __int64 v9; // r13
  char v10; // cl
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 *v14; // r15
  __int64 v15; // r14
  __int64 *v16; // r12
  __int64 v17; // r15
  _QWORD *v18; // rax
  char v19; // al
  __int64 v20; // r14
  unsigned int v21; // eax
  __int64 v22; // rdx
  char v23; // cl
  _BYTE *v24; // rax
  char v25; // cl
  __int64 *v26; // rdi
  __int64 v27; // rbx
  __int64 v28; // r12
  __int64 v29; // rdx
  unsigned int v30; // esi
  __int64 v31; // r14
  int v32; // r14d
  unsigned int v33; // r15d
  _BYTE *v34; // rax
  __int64 v35; // rdx
  int v36; // r12d
  __int64 v37; // r12
  __int64 v38; // r15
  __int64 v39; // rdx
  unsigned int v40; // esi
  unsigned __int64 v41; // [rsp+8h] [rbp-A8h]
  __int64 v42; // [rsp+10h] [rbp-A0h]
  char v43; // [rsp+1Bh] [rbp-95h]
  char v44; // [rsp+1Bh] [rbp-95h]
  char v45; // [rsp+1Bh] [rbp-95h]
  unsigned int v46; // [rsp+1Ch] [rbp-94h]
  int v47[8]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v48; // [rsp+40h] [rbp-70h]
  __int64 v49; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v50; // [rsp+58h] [rbp-58h]
  __int16 v51; // [rsp+70h] [rbp-40h]

  v2 = *(unsigned __int8 **)(a2 - 64);
  v3 = *((_QWORD *)v2 + 2);
  if ( !v3 )
    return 0;
  v4 = *(_QWORD **)(v3 + 8);
  if ( v4 )
    return 0;
  v6 = *v2;
  v8 = *(_BYTE **)(a2 - 32);
  if ( *v2 != 42 && v6 != 46 && v6 != 55 && v6 != 54 )
  {
    if ( v6 == 44 )
    {
      v41 = *((_QWORD *)v2 - 8);
      if ( *(_BYTE *)v41 <= 0x15u && v8 == *((_BYTE **)v2 - 4) )
        goto LABEL_12;
    }
    return 0;
  }
  if ( v8 != *((_BYTE **)v2 - 8) )
    return 0;
  v41 = *((_QWORD *)v2 - 4);
  if ( *(_BYTE *)v41 > 0x15u )
    return 0;
LABEL_12:
  if ( *v8 != 68 )
    return v4;
  v9 = *((_QWORD *)v8 - 4);
  if ( !v9 )
    return v4;
  v10 = sub_BD3660((__int64)v8, 3);
  if ( v10 )
    return v4;
  v42 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v42 + 8) - 17 > 1 )
  {
    v19 = sub_F0C890(a1, v42, *(_QWORD *)(v9 + 8));
    v10 = 0;
    if ( !v19 )
      return v4;
  }
  v11 = *v2;
  v46 = v11 - 29;
  if ( (unsigned int)(v11 - 54) > 1 )
    goto LABEL_17;
  v43 = v10;
  v20 = (unsigned int)sub_BCB060(*(_QWORD *)(v9 + 8));
  v21 = sub_BCB060(*(_QWORD *)(v41 + 8));
  v23 = v43;
  v50 = v21;
  if ( v21 > 0x40 )
  {
    sub_C43690((__int64)&v49, v20, 0);
    v23 = v43;
  }
  else
  {
    v49 = v20;
  }
  v24 = (_BYTE *)v41;
  if ( *(_BYTE *)v41 != 17 )
  {
    v44 = v23;
    v31 = *(_QWORD *)(v41 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v31 + 8) - 17 > 1 )
      goto LABEL_39;
    v24 = sub_AD7630(v41, 0, v22);
    v25 = v44;
    if ( !v24 || *v24 != 17 )
    {
      if ( *(_BYTE *)(v31 + 8) == 17 )
      {
        v32 = *(_DWORD *)(v31 + 32);
        if ( v32 )
        {
          v33 = 0;
          while ( 1 )
          {
            v45 = v25;
            v34 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v41, v33);
            if ( !v34 )
              break;
            v25 = v45;
            if ( *v34 != 13 )
            {
              if ( *v34 != 17 )
                break;
              v25 = sub_B532C0((__int64)(v34 + 24), &v49, 36);
              if ( !v25 )
                break;
            }
            if ( v32 == ++v33 )
              goto LABEL_31;
          }
        }
      }
      goto LABEL_39;
    }
  }
  v25 = sub_B532C0((__int64)(v24 + 24), &v49, 36);
LABEL_31:
  if ( v25 )
  {
    if ( v50 > 0x40 && v49 )
      j_j___libc_free_0_0(v49);
LABEL_17:
    v12 = sub_AD4C30(v41, *(__int64 ***)(v9 + 8), 0);
    v13 = v12;
    if ( v46 == 15 )
    {
      v26 = *(__int64 **)(a1 + 32);
      v51 = 257;
      v15 = sub_10BBE20(v26, 15, v12, v9, v47[0], 0, (__int64)&v49, 0);
    }
    else
    {
      v14 = *(__int64 **)(a1 + 32);
      v48 = 257;
      v15 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v14[10] + 16LL))(
              v14[10],
              v46,
              v9,
              v12);
      if ( !v15 )
      {
        v51 = 257;
        v15 = sub_B504D0(v46, v9, v13, (__int64)&v49, 0, 0);
        if ( (unsigned __int8)sub_920620(v15) )
        {
          v35 = v14[12];
          v36 = *((_DWORD *)v14 + 26);
          if ( v35 )
            sub_B99FD0(v15, 3u, v35);
          sub_B45150(v15, v36);
        }
        (*(void (__fastcall **)(__int64, __int64, int *, __int64, __int64))(*(_QWORD *)v14[11] + 16LL))(
          v14[11],
          v15,
          v47,
          v14[7],
          v14[8]);
        v37 = *v14;
        v38 = *v14 + 16LL * *((unsigned int *)v14 + 2);
        while ( v38 != v37 )
        {
          v39 = *(_QWORD *)(v37 + 8);
          v40 = *(_DWORD *)v37;
          v37 += 16;
          sub_B99FD0(v15, v40, v39);
        }
      }
    }
    v16 = *(__int64 **)(a1 + 32);
    v48 = 257;
    v17 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v16[10] + 16LL))(
            v16[10],
            28,
            v15,
            v9);
    if ( !v17 )
    {
      v51 = 257;
      v17 = sub_B504D0(28, v15, v9, (__int64)&v49, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, int *, __int64, __int64))(*(_QWORD *)v16[11] + 16LL))(
        v16[11],
        v17,
        v47,
        v16[7],
        v16[8]);
      v27 = *v16;
      v28 = *v16 + 16LL * *((unsigned int *)v16 + 2);
      while ( v28 != v27 )
      {
        v29 = *(_QWORD *)(v27 + 8);
        v30 = *(_DWORD *)v27;
        v27 += 16;
        sub_B99FD0(v17, v30, v29);
      }
    }
    v51 = 257;
    v18 = sub_BD2C40(72, unk_3F10A14);
    v4 = v18;
    if ( v18 )
      sub_B515B0((__int64)v18, v17, v42, (__int64)&v49, 0, 0);
    return v4;
  }
LABEL_39:
  if ( v50 > 0x40 && v49 )
    j_j___libc_free_0_0(v49);
  return v4;
}
