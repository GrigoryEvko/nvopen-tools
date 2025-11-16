// Function: sub_9D23D0
// Address: 0x9d23d0
//
__int64 *__fastcall sub_9D23D0(__int64 *a1, _QWORD *a2)
{
  char v4; // dl
  char v5; // al
  _QWORD *v7; // rsi
  __int64 v8; // rax
  _QWORD *v9; // rdi
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // rdx
  _QWORD *v13; // r15
  __int64 v14; // rbx
  __int64 v15; // r13
  __int64 v16; // rdi
  _QWORD *v17; // rdi
  __int64 v18; // r13
  __int64 v19; // rbx
  volatile signed __int32 *v20; // rdi
  signed __int32 v21; // eax
  void (*v22)(void); // rax
  signed __int32 v23; // eax
  void (*v24)(void); // rcx
  bool v25; // zf
  __int64 v26; // rax
  __int64 v27; // [rsp+0h] [rbp-D0h]
  _QWORD *v28; // [rsp+18h] [rbp-B8h]
  _QWORD v29[3]; // [rsp+20h] [rbp-B0h] BYREF
  char v30; // [rsp+38h] [rbp-98h]
  __int64 v31; // [rsp+40h] [rbp-90h] BYREF
  __int64 v32; // [rsp+48h] [rbp-88h]
  __int64 v33; // [rsp+50h] [rbp-80h]
  char v34; // [rsp+58h] [rbp-78h]
  char v35; // [rsp+60h] [rbp-70h]
  const char *v36; // [rsp+70h] [rbp-60h] BYREF
  char v37; // [rsp+90h] [rbp-40h]
  char v38; // [rsp+91h] [rbp-3Fh]

  sub_A4E6C0(&v31, a2 + 3, 0);
  v4 = v35 & 1;
  v5 = (2 * (v35 & 1)) | v35 & 0xFD;
  v35 = v5;
  if ( v4 )
  {
    v35 = v5 & 0xFD;
    v26 = v31;
    v31 = 0;
    *a1 = v26 | 1;
    goto LABEL_44;
  }
  v30 = 0;
  if ( !v34 )
  {
    v38 = 1;
    v37 = 3;
    v36 = "Malformed block";
    sub_9C81F0(a1, (__int64)a2, (__int64)&v36);
    goto LABEL_4;
  }
  v7 = (_QWORD *)*a2;
  v8 = v33;
  v30 = 1;
  v9 = (_QWORD *)a2[1];
  v10 = a2[2];
  v33 = 0;
  v11 = v31;
  v12 = v32;
  a2[2] = v8;
  v13 = v7;
  v32 = 0;
  v31 = 0;
  v28 = v9;
  v27 = v10;
  *a2 = v11;
  a2[1] = v12;
  memset(v29, 0, sizeof(v29));
  if ( v7 != v9 )
  {
    while ( 1 )
    {
      v14 = v13[9];
      v15 = v13[8];
      if ( v14 != v15 )
      {
        do
        {
          v16 = *(_QWORD *)(v15 + 8);
          if ( v16 != v15 + 24 )
            j_j___libc_free_0(v16, *(_QWORD *)(v15 + 24) + 1LL);
          v15 += 40;
        }
        while ( v14 != v15 );
        v15 = v13[8];
      }
      if ( v15 )
        j_j___libc_free_0(v15, v13[10] - v15);
      v17 = (_QWORD *)v13[4];
      if ( v17 != v13 + 6 )
        j_j___libc_free_0(v17, v13[6] + 1LL);
      v18 = v13[2];
      v19 = v13[1];
      if ( v18 != v19 )
        break;
LABEL_34:
      if ( v19 )
        j_j___libc_free_0(v19, v13[3] - v19);
      v13 += 11;
      if ( v28 == v13 )
        goto LABEL_37;
    }
    while ( 1 )
    {
      v20 = *(volatile signed __int32 **)(v19 + 8);
      if ( !v20 )
        goto LABEL_21;
      if ( &_pthread_key_create )
      {
        v21 = _InterlockedExchangeAdd(v20 + 2, 0xFFFFFFFF);
      }
      else
      {
        v21 = *((_DWORD *)v20 + 2);
        *((_DWORD *)v20 + 2) = v21 - 1;
      }
      if ( v21 != 1 )
        goto LABEL_21;
      v22 = *(void (**)(void))(*(_QWORD *)v20 + 16LL);
      if ( v22 != nullsub_25 )
        v22();
      if ( &_pthread_key_create )
      {
        v23 = _InterlockedExchangeAdd(v20 + 3, 0xFFFFFFFF);
      }
      else
      {
        v23 = *((_DWORD *)v20 + 3);
        *((_DWORD *)v20 + 3) = v23 - 1;
      }
      if ( v23 != 1 )
        goto LABEL_21;
      v24 = *(void (**)(void))(*(_QWORD *)v20 + 24LL);
      if ( (char *)v24 == (char *)sub_9C26E0 )
      {
        (*(void (**)(void))(*(_QWORD *)v20 + 8LL))();
        v19 += 16;
        if ( v18 == v19 )
        {
LABEL_33:
          v19 = v13[1];
          goto LABEL_34;
        }
      }
      else
      {
        v24();
LABEL_21:
        v19 += 16;
        if ( v18 == v19 )
          goto LABEL_33;
      }
    }
  }
LABEL_37:
  if ( v7 )
    j_j___libc_free_0(v7, v27 - (_QWORD)v7);
  v25 = v30 == 0;
  *a1 = 1;
  if ( !v25 )
  {
    v30 = 0;
    sub_9C90D0(v29);
  }
LABEL_4:
  if ( (v35 & 2) != 0 )
    sub_9D2360(&v31);
  if ( (v35 & 1) == 0 )
  {
    if ( v34 )
    {
      v34 = 0;
      sub_9C90D0(&v31);
    }
    return a1;
  }
LABEL_44:
  if ( v31 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v31 + 8LL))(v31);
  return a1;
}
