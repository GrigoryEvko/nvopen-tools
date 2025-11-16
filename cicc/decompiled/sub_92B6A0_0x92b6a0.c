// Function: sub_92B6A0
// Address: 0x92b6a0
//
_BYTE *__fastcall sub_92B6A0(__int64 *a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  __int64 v8; // rsi
  int v9; // ecx
  unsigned __int8 v10; // al
  unsigned int **v11; // r15
  unsigned int *v12; // rdi
  unsigned int v13; // ebx
  __int64 (__fastcall *v14)(__int64, unsigned int, _BYTE *, _BYTE *); // rax
  _BYTE *v15; // r12
  __int64 v16; // rax
  unsigned int *v17; // rdx
  unsigned int *v18; // r13
  __int64 v19; // r14
  __int64 v20; // rdx
  __int64 v21; // rsi
  int v22; // eax
  unsigned int **v23; // r15
  unsigned int *v24; // rdi
  __int64 (__fastcall *v25)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8); // rax
  unsigned int *v26; // r13
  __int64 v27; // r14
  __int64 v28; // rdx
  __int64 v29; // rsi
  unsigned int **v30; // r15
  unsigned int *v31; // rdi
  __int64 (__fastcall *v32)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8); // rax
  __int64 v34; // rsi
  char v35; // bl
  unsigned int *v36; // r13
  __int64 v37; // r14
  __int64 v38; // rdx
  __int64 v39; // rsi
  __int64 v40; // rax
  unsigned int **v41; // r13
  __int64 v42; // rax
  unsigned __int64 v43; // rsi
  __int64 v44; // [rsp+8h] [rbp-A8h]
  int v45; // [rsp+10h] [rbp-A0h]
  _QWORD v46[4]; // [rsp+20h] [rbp-90h] BYREF
  char v47; // [rsp+40h] [rbp-70h]
  char v48; // [rsp+41h] [rbp-6Fh]
  _BYTE v49[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v50; // [rsp+70h] [rbp-40h]

  v8 = *(_QWORD *)(a2 + 8);
  v9 = *(unsigned __int8 *)(v8 + 8);
  v10 = *(_BYTE *)(v8 + 8);
  if ( (unsigned int)(v9 - 17) <= 1 )
    v10 = *(_BYTE *)(**(_QWORD **)(v8 + 16) + 8LL);
  if ( v10 > 3u && v10 != 5 && (v10 & 0xFD) != 4 )
  {
    if ( (unsigned __int8)sub_91B6F0(a4) )
    {
      v23 = (unsigned int **)a1[1];
      v48 = 1;
      v46[0] = "div";
      v47 = 3;
      v24 = v23[10];
      v25 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8))(*(_QWORD *)v24 + 24LL);
      if ( v25 == sub_920250 )
      {
        if ( *(_BYTE *)a2 > 0x15u || *a3 > 0x15u )
        {
LABEL_30:
          v50 = 257;
          v15 = (_BYTE *)sub_B504D0(20, a2, a3, v49, 0, 0);
          (*(void (__fastcall **)(unsigned int *, _BYTE *, _QWORD *, unsigned int *, unsigned int *))(*(_QWORD *)v23[11] + 16LL))(
            v23[11],
            v15,
            v46,
            v23[7],
            v23[8]);
          v26 = *v23;
          v27 = (__int64)&(*v23)[4 * *((unsigned int *)v23 + 2)];
          if ( *v23 != (unsigned int *)v27 )
          {
            do
            {
              v28 = *((_QWORD *)v26 + 1);
              v29 = *v26;
              v26 += 4;
              sub_B99FD0(v15, v29, v28);
            }
            while ( (unsigned int *)v27 != v26 );
          }
          return v15;
        }
        if ( (unsigned __int8)sub_AC47B0(20) )
          v15 = (_BYTE *)sub_AD5570(20, a2, a3, 0, 0);
        else
          v15 = (_BYTE *)sub_AABE40(20, a2, a3);
      }
      else
      {
        v15 = (_BYTE *)v25((__int64)v24, 20u, (_BYTE *)a2, a3, 0);
      }
      if ( v15 )
        return v15;
      goto LABEL_30;
    }
    v30 = (unsigned int **)a1[1];
    v48 = 1;
    v46[0] = "div";
    v47 = 3;
    v31 = v30[10];
    v32 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8))(*(_QWORD *)v31 + 24LL);
    if ( v32 == sub_920250 )
    {
      if ( *(_BYTE *)a2 > 0x15u || *a3 > 0x15u )
      {
LABEL_46:
        v50 = 257;
        v15 = (_BYTE *)sub_B504D0(19, a2, a3, v49, 0, 0);
        (*(void (__fastcall **)(unsigned int *, _BYTE *, _QWORD *, unsigned int *, unsigned int *))(*(_QWORD *)v30[11]
                                                                                                  + 16LL))(
          v30[11],
          v15,
          v46,
          v30[7],
          v30[8]);
        v36 = *v30;
        v37 = (__int64)&(*v30)[4 * *((unsigned int *)v30 + 2)];
        if ( *v30 != (unsigned int *)v37 )
        {
          do
          {
            v38 = *((_QWORD *)v36 + 1);
            v39 = *v36;
            v36 += 4;
            sub_B99FD0(v15, v39, v38);
          }
          while ( (unsigned int *)v37 != v36 );
        }
        return v15;
      }
      if ( (unsigned __int8)sub_AC47B0(19) )
        v15 = (_BYTE *)sub_AD5570(19, a2, a3, 0, 0);
      else
        v15 = (_BYTE *)sub_AABE40(19, a2, a3);
    }
    else
    {
      v15 = (_BYTE *)v32((__int64)v31, 19u, (_BYTE *)a2, a3, 0);
    }
    if ( v15 )
      return v15;
    goto LABEL_46;
  }
  if ( *(_BYTE *)a2 == 18 )
  {
    if ( (_BYTE)v9 != 2 )
      goto LABEL_6;
    v44 = sub_C33320(a4);
    sub_C3B1B0(v49, 1.0);
    sub_C407B0(v46, v49, v44);
    sub_C338F0(v49);
    sub_C41640(v46, *(_QWORD *)(a2 + 24), 1, v49);
    v35 = sub_AC3090(a2, v46);
    sub_91D830(v46);
    if ( v35 )
      goto LABEL_6;
  }
  else if ( (_BYTE)v9 != 2 )
  {
    goto LABEL_6;
  }
  if ( !unk_4D0451C )
  {
    if ( unk_4D04518 )
    {
      v34 = 8493;
      goto LABEL_52;
    }
LABEL_6:
    v11 = (unsigned int **)a1[1];
    v48 = 1;
    v46[0] = "div";
    v47 = 3;
    if ( *((_BYTE *)v11 + 108) )
    {
      v15 = (_BYTE *)sub_B35400((_DWORD)v11, 105, a2, (_DWORD)a3, v45, (__int64)v46, 0, 0);
      goto LABEL_18;
    }
    v12 = v11[10];
    v13 = *((_DWORD *)v11 + 26);
    v14 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *))(*(_QWORD *)v12 + 40LL);
    if ( v14 == sub_928A40 )
    {
      if ( *(_BYTE *)a2 > 0x15u || *a3 > 0x15u )
      {
LABEL_14:
        v50 = 257;
        v16 = sub_B504D0(21, a2, a3, v49, 0, 0);
        v17 = v11[12];
        v15 = (_BYTE *)v16;
        if ( v17 )
          sub_B99FD0(v16, 3, v17);
        sub_B45150(v15, v13);
        (*(void (__fastcall **)(unsigned int *, _BYTE *, _QWORD *, unsigned int *, unsigned int *))(*(_QWORD *)v11[11]
                                                                                                  + 16LL))(
          v11[11],
          v15,
          v46,
          v11[7],
          v11[8]);
        v18 = *v11;
        v19 = (__int64)&(*v11)[4 * *((unsigned int *)v11 + 2)];
        if ( *v11 != (unsigned int *)v19 )
        {
          do
          {
            v20 = *((_QWORD *)v18 + 1);
            v21 = *v18;
            v18 += 4;
            sub_B99FD0(v15, v21, v20);
          }
          while ( (unsigned int *)v19 != v18 );
        }
LABEL_18:
        if ( !unk_4D04700 )
          return v15;
        goto LABEL_19;
      }
      if ( (unsigned __int8)sub_AC47B0(21) )
        v15 = (_BYTE *)sub_AD5570(21, a2, a3, 0, 0);
      else
        v15 = (_BYTE *)sub_AABE40(21, a2, a3);
    }
    else
    {
      v15 = (_BYTE *)((__int64 (__fastcall *)(unsigned int *, __int64, __int64, _BYTE *, _QWORD))v14)(
                       v12,
                       21,
                       a2,
                       a3,
                       v13);
    }
    if ( v15 )
      goto LABEL_18;
    v13 = *((_DWORD *)v11 + 26);
    goto LABEL_14;
  }
  v34 = unk_4D04518 == 0 ? 8491 : 8493;
LABEL_52:
  v40 = *a1;
  v46[0] = a2;
  v41 = (unsigned int **)a1[1];
  v46[1] = a3;
  v50 = 257;
  v42 = sub_B6E160(**(_QWORD **)(v40 + 32), v34, 0, 0);
  v43 = 0;
  if ( v42 )
    v43 = *(_QWORD *)(v42 + 24);
  v15 = (_BYTE *)sub_921880(v41, v43, v42, (int)v46, 2, (__int64)v49, 0);
  if ( unk_4D04700 )
  {
LABEL_19:
    if ( *v15 > 0x1Cu )
    {
      v22 = sub_B45210(v15);
      sub_B45150(v15, v22 | 1u);
    }
  }
  return v15;
}
