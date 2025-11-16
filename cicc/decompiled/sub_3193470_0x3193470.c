// Function: sub_3193470
// Address: 0x3193470
//
__int64 __fastcall sub_3193470(__int64 a1, __int64 a2, char a3)
{
  _QWORD *v3; // r14
  __int64 v4; // rsi
  _QWORD *i; // r13
  __int64 v6; // rsi
  __int64 v7; // rbx
  _QWORD *v8; // rbx
  int v9; // eax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdi
  int v13; // eax
  unsigned int v14; // ecx
  int *v15; // r10
  int v16; // esi
  unsigned int v17; // r14d
  _QWORD *v18; // rax
  __int64 v19; // rax
  unsigned int v20; // eax
  _QWORD *v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rax
  _QWORD *v25; // r12
  _QWORD *v26; // rbx
  char *v27; // rdi
  char **v28; // r14
  int v29; // r10d
  int v30; // r11d
  unsigned __int8 v32; // [rsp+17h] [rbp-D9h]
  __int64 v34; // [rsp+20h] [rbp-D0h] BYREF
  _QWORD *v35; // [rsp+28h] [rbp-C8h]
  __int64 v36; // [rsp+30h] [rbp-C0h]
  unsigned int v37; // [rsp+38h] [rbp-B8h]
  char *v38; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v39; // [rsp+48h] [rbp-A8h] BYREF
  __int64 v40; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v41; // [rsp+58h] [rbp-98h]
  _QWORD v42[2]; // [rsp+60h] [rbp-90h] BYREF
  __int64 v43; // [rsp+70h] [rbp-80h]
  __int64 v44; // [rsp+80h] [rbp-70h] BYREF
  _QWORD *v45; // [rsp+88h] [rbp-68h] BYREF
  void (__fastcall *v46)(__int64 *, __int64 *, __int64); // [rsp+90h] [rbp-60h]
  __int64 v47; // [rsp+98h] [rbp-58h]
  _QWORD v48[2]; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v49; // [rsp+B0h] [rbp-40h]

  v3 = *(_QWORD **)(a1 + 56);
  v4 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v32 = 0;
  if ( !v3 )
    goto LABEL_43;
  for ( i = v3 - 3; ; i = v8 )
  {
    v7 = i[4];
    if ( v7 == i[5] + 48LL || !v7 )
      v8 = 0;
    else
      v8 = (_QWORD *)(v7 - 24);
    if ( (unsigned __int8)sub_BD3610((__int64)i, 0) )
      goto LABEL_6;
    LOBYTE(v44) = 0;
    v45 = 0;
    v46 = 0;
    v47 = 0;
    LOBYTE(v48[0]) = a3;
    v9 = *(unsigned __int8 *)i;
    if ( (unsigned int)(v9 - 29) <= 0x14 )
    {
      if ( (unsigned int)(v9 - 29) > 0x12 )
        goto LABEL_14;
    }
    else if ( (unsigned int)(v9 - 51) <= 1 )
    {
LABEL_14:
      v45 = i;
      v10 = i[1];
      if ( *(_BYTE *)(v10 + 8) == 12 )
      {
        v11 = *(unsigned int *)(a2 + 24);
        v12 = *(_QWORD *)(a2 + 8);
        v13 = *(_DWORD *)(v10 + 8) >> 8;
        if ( (_DWORD)v11 )
        {
          v14 = (v11 - 1) & (37 * v13);
          v15 = (int *)(v12 + 8LL * v14);
          v16 = *v15;
          if ( *v15 == v13 )
          {
LABEL_17:
            if ( v15 != (int *)(v12 + 8 * v11) )
            {
              v17 = v15[1];
              v18 = (_QWORD *)sub_BD5C60((__int64)i);
              v46 = (void (__fastcall *)(__int64 *, __int64 *, __int64))sub_BCCE00(v18, v17);
              v19 = i[5];
              LOBYTE(v44) = 1;
              v47 = v19;
            }
          }
          else
          {
            v29 = 1;
            while ( v16 != -1 )
            {
              v30 = v29 + 1;
              v14 = (v11 - 1) & (v29 + v14);
              v15 = (int *)(v12 + 8LL * v14);
              v16 = *v15;
              if ( v13 == *v15 )
                goto LABEL_17;
              v29 = v30;
            }
          }
        }
      }
    }
    v6 = sub_3190D00((__int64)&v44, (__int64)&v34);
    if ( v6 )
    {
      sub_BD84D0((__int64)i, v6);
      sub_B43D60(i);
      v32 = 1;
    }
LABEL_6:
    if ( !v8 )
      break;
  }
  v3 = v35;
  v20 = v37;
  v4 = 9LL * v37;
  if ( (_DWORD)v36 )
  {
    v25 = &v35[v4];
    if ( &v35[v4] != v35 )
    {
      v26 = v35;
      while ( !v26[3] && !v26[6] )
      {
        v26 += 9;
        if ( v25 == v26 )
          goto LABEL_21;
      }
      if ( v25 != v26 )
      {
        do
        {
          v27 = (char *)v26[7];
          v28 = &v38;
          v38 = v27;
          v39 = v26[8];
          while ( 1 )
          {
            v46 = 0;
            sub_F5CAB0(v27, 0, 0, (__int64)&v44);
            if ( v46 )
              v46(&v44, &v44, 3);
            if ( &v40 == (__int64 *)++v28 )
              break;
            v27 = *v28;
          }
          v26 += 9;
          if ( v26 == v25 )
            break;
          while ( !v26[3] && !v26[6] )
          {
            v26 += 9;
            if ( v25 == v26 )
              goto LABEL_56;
          }
        }
        while ( v25 != v26 );
LABEL_56:
        v3 = v35;
        v20 = v37;
        v4 = 9LL * v37;
      }
    }
  }
LABEL_21:
  if ( v20 )
  {
    v38 = 0;
    v21 = &v3[v4];
    v39 = 0;
    v40 = 0;
    v41 = 0;
    v42[0] = 0;
    v42[1] = 0;
    v43 = 0;
    v44 = 1;
    v45 = 0;
    v46 = 0;
    v47 = 0;
    v48[0] = 0;
    v48[1] = 0;
    v49 = 0;
    do
    {
      v22 = v3[6];
      if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
        sub_BD60C0(v3 + 4);
      v23 = v3[3];
      if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
        sub_BD60C0(v3 + 1);
      v3 += 9;
    }
    while ( v21 != v3 );
    if ( v49 != -4096 && v49 != 0 && v49 != -8192 )
      sub_BD60C0(v48);
    if ( v47 != -4096 && v47 != 0 && v47 != -8192 )
      sub_BD60C0(&v45);
    if ( v43 != 0 && v43 != -4096 && v43 != -8192 )
      sub_BD60C0(v42);
    if ( v41 != -4096 && v41 != 0 && v41 != -8192 )
      sub_BD60C0(&v39);
    v3 = v35;
    v4 = 9LL * v37;
  }
LABEL_43:
  sub_C7D6A0((__int64)v3, v4 * 8, 8);
  return v32;
}
