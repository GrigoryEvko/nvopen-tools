// Function: sub_929130
// Address: 0x929130
//
_BYTE *__fastcall sub_929130(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  __int64 v7; // rcx
  int v8; // eax
  unsigned int **v9; // r15
  unsigned int *v10; // rdi
  unsigned int v11; // ebx
  __int64 (__fastcall *v12)(__int64, unsigned int, _BYTE *, _BYTE *); // rax
  _BYTE *v13; // r12
  int v14; // eax
  unsigned int **v16; // r15
  unsigned int *v17; // rdi
  __int64 (__fastcall *v18)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char); // rax
  unsigned int *v19; // rbx
  __int64 v20; // r13
  __int64 v21; // rdx
  __int64 v22; // rsi
  unsigned int **v23; // r15
  unsigned int *v24; // rdi
  __int64 (__fastcall *v25)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char); // rax
  unsigned int *v26; // rbx
  __int64 v27; // r13
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // rax
  unsigned int *v31; // rdx
  unsigned int *v32; // rbx
  __int64 v33; // r13
  __int64 v34; // rdx
  __int64 v35; // rsi
  int v36; // [rsp+0h] [rbp-A0h]
  _QWORD v37[4]; // [rsp+10h] [rbp-90h] BYREF
  char v38; // [rsp+30h] [rbp-70h]
  char v39; // [rsp+31h] [rbp-6Fh]
  _BYTE v40[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v41; // [rsp+60h] [rbp-40h]

  v7 = *(_QWORD *)(a2 + 8);
  v8 = *(unsigned __int8 *)(v7 + 8);
  if ( (unsigned int)(v8 - 17) <= 1 )
    LOBYTE(v8) = *(_BYTE *)(**(_QWORD **)(v7 + 16) + 8LL);
  if ( (unsigned __int8)v8 > 3u && (_BYTE)v8 != 5 && (v8 & 0xFD) != 4 )
  {
    if ( (unsigned __int8)sub_91B6F0(a4) )
    {
      v16 = *(unsigned int ***)(a1 + 8);
      v39 = 1;
      v37[0] = "mul";
      v38 = 3;
      v17 = v16[10];
      v18 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char))(*(_QWORD *)v17 + 32LL);
      if ( v18 == sub_9201A0 )
      {
        if ( *(_BYTE *)a2 > 0x15u || *a3 > 0x15u )
        {
LABEL_24:
          v41 = 257;
          v13 = (_BYTE *)sub_B504D0(17, a2, a3, v40, 0, 0);
          (*(void (__fastcall **)(unsigned int *, _BYTE *, _QWORD *, unsigned int *, unsigned int *))(*(_QWORD *)v16[11] + 16LL))(
            v16[11],
            v13,
            v37,
            v16[7],
            v16[8]);
          v19 = *v16;
          v20 = (__int64)&(*v16)[4 * *((unsigned int *)v16 + 2)];
          if ( *v16 != (unsigned int *)v20 )
          {
            do
            {
              v21 = *((_QWORD *)v19 + 1);
              v22 = *v19;
              v19 += 4;
              sub_B99FD0(v13, v22, v21);
            }
            while ( (unsigned int *)v20 != v19 );
          }
          sub_B44850(v13, 1);
          return v13;
        }
        if ( (unsigned __int8)sub_AC47B0(17) )
          v13 = (_BYTE *)sub_AD5570(17, a2, a3, 2, 0);
        else
          v13 = (_BYTE *)sub_AABE40(17, a2, a3);
      }
      else
      {
        v13 = (_BYTE *)v18((__int64)v17, 17u, (_BYTE *)a2, a3, 0, 1);
      }
      if ( v13 )
        return v13;
      goto LABEL_24;
    }
    v23 = *(unsigned int ***)(a1 + 8);
    v39 = 1;
    v37[0] = "mul";
    v38 = 3;
    v24 = v23[10];
    v25 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char))(*(_QWORD *)v24 + 32LL);
    if ( v25 == sub_9201A0 )
    {
      if ( *(_BYTE *)a2 > 0x15u || *a3 > 0x15u )
      {
LABEL_33:
        v41 = 257;
        v13 = (_BYTE *)sub_B504D0(17, a2, a3, v40, 0, 0);
        (*(void (__fastcall **)(unsigned int *, _BYTE *, _QWORD *, unsigned int *, unsigned int *))(*(_QWORD *)v23[11]
                                                                                                  + 16LL))(
          v23[11],
          v13,
          v37,
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
            sub_B99FD0(v13, v29, v28);
          }
          while ( (unsigned int *)v27 != v26 );
        }
        return v13;
      }
      if ( (unsigned __int8)sub_AC47B0(17) )
        v13 = (_BYTE *)sub_AD5570(17, a2, a3, 0, 0);
      else
        v13 = (_BYTE *)sub_AABE40(17, a2, a3);
    }
    else
    {
      v13 = (_BYTE *)v25((__int64)v24, 17u, (_BYTE *)a2, a3, 0, 0);
    }
    if ( v13 )
      return v13;
    goto LABEL_33;
  }
  v9 = *(unsigned int ***)(a1 + 8);
  v39 = 1;
  v37[0] = "mul";
  v38 = 3;
  if ( *((_BYTE *)v9 + 108) )
  {
    v13 = (_BYTE *)sub_B35400((_DWORD)v9, 108, a2, (_DWORD)a3, v36, (__int64)v37, 0, 0);
    goto LABEL_11;
  }
  v10 = v9[10];
  v11 = *((_DWORD *)v9 + 26);
  v12 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *))(*(_QWORD *)v10 + 40LL);
  if ( v12 == sub_928A40 )
  {
    if ( *(_BYTE *)a2 > 0x15u || *a3 > 0x15u )
      goto LABEL_38;
    if ( (unsigned __int8)sub_AC47B0(18) )
      v13 = (_BYTE *)sub_AD5570(18, a2, a3, 0, 0);
    else
      v13 = (_BYTE *)sub_AABE40(18, a2, a3);
  }
  else
  {
    v13 = (_BYTE *)((__int64 (__fastcall *)(unsigned int *, __int64, __int64, _BYTE *, _QWORD))v12)(
                     v10,
                     18,
                     a2,
                     a3,
                     v11);
  }
  if ( !v13 )
  {
    v11 = *((_DWORD *)v9 + 26);
LABEL_38:
    v41 = 257;
    v30 = sub_B504D0(18, a2, a3, v40, 0, 0);
    v31 = v9[12];
    v13 = (_BYTE *)v30;
    if ( v31 )
      sub_B99FD0(v30, 3, v31);
    sub_B45150(v13, v11);
    (*(void (__fastcall **)(unsigned int *, _BYTE *, _QWORD *, unsigned int *, unsigned int *))(*(_QWORD *)v9[11] + 16LL))(
      v9[11],
      v13,
      v37,
      v9[7],
      v9[8]);
    v32 = *v9;
    v33 = (__int64)&(*v9)[4 * *((unsigned int *)v9 + 2)];
    if ( *v9 != (unsigned int *)v33 )
    {
      do
      {
        v34 = *((_QWORD *)v32 + 1);
        v35 = *v32;
        v32 += 4;
        sub_B99FD0(v13, v35, v34);
      }
      while ( (unsigned int *)v33 != v32 );
    }
  }
LABEL_11:
  if ( unk_4D04700 && *v13 > 0x1Cu )
  {
    v14 = sub_B45210(v13);
    sub_B45150(v13, v14 | 1u);
  }
  return v13;
}
