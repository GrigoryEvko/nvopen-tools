// Function: sub_2C837C0
// Address: 0x2c837c0
//
void __fastcall sub_2C837C0(
        unsigned __int64 *a1,
        __int64 a2,
        int a3,
        int a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  char *v9; // r14
  char *v10; // rbx
  __int64 v11; // rdi
  __int64 v12; // rax
  char *v13; // rdi
  __int64 v14; // rax
  unsigned __int64 v15; // r11
  void (*v16)(); // rax
  unsigned __int64 *v17; // rsi
  bool v22; // [rsp+28h] [rbp-48h]
  unsigned __int64 v23; // [rsp+28h] [rbp-48h]
  unsigned __int64 v24[7]; // [rsp+38h] [rbp-38h] BYREF

  v9 = (char *)a1[2];
  v10 = (char *)a1[1];
  v11 = (v9 - v10) >> 5;
  v12 = (v9 - v10) >> 3;
  if ( v11 > 0 )
  {
    v13 = &v10[32 * v11];
    while ( a2 != **(_QWORD **)v10 )
    {
      if ( a2 == **((_QWORD **)v10 + 1) )
      {
        v10 += 8;
        v22 = v9 == v10;
        goto LABEL_9;
      }
      if ( a2 == **((_QWORD **)v10 + 2) )
      {
        v10 += 16;
        v22 = v9 == v10;
        goto LABEL_9;
      }
      if ( a2 == **((_QWORD **)v10 + 3) )
      {
        v10 += 24;
        v22 = v9 == v10;
        goto LABEL_9;
      }
      v10 += 32;
      if ( v13 == v10 )
      {
        v12 = (v9 - v10) >> 3;
        goto LABEL_21;
      }
    }
    goto LABEL_8;
  }
LABEL_21:
  if ( v12 == 2 )
  {
LABEL_27:
    if ( a2 == **(_QWORD **)v10 )
      goto LABEL_8;
    v10 += 8;
    goto LABEL_29;
  }
  if ( v12 == 3 )
  {
    if ( a2 == **(_QWORD **)v10 )
      goto LABEL_8;
    v10 += 8;
    goto LABEL_27;
  }
  if ( v12 != 1 )
  {
LABEL_24:
    v22 = 1;
    v10 = v9;
    goto LABEL_9;
  }
LABEL_29:
  if ( a2 != **(_QWORD **)v10 )
    goto LABEL_24;
LABEL_8:
  v22 = v9 == v10;
LABEL_9:
  v14 = sub_22077B0(0x38u);
  v15 = v14;
  if ( v14 )
  {
    *(_QWORD *)v14 = a2;
    *(_QWORD *)(v14 + 40) = a8;
    *(_DWORD *)(v14 + 8) = a3;
    *(_DWORD *)(v14 + 12) = a4;
    *(_QWORD *)(v14 + 16) = a5;
    *(_QWORD *)(v14 + 24) = a6;
    *(_QWORD *)(v14 + 32) = a7;
    *(_BYTE *)(v14 + 48) = v22;
    if ( v9 == v10 )
    {
      v16 = *(void (**)())(*(_QWORD *)a2 + 88LL);
      if ( v16 != nullsub_19 )
      {
        v23 = v15;
        ((void (__fastcall *)(__int64))v16)(a2);
        v15 = v23;
      }
    }
  }
  v24[0] = v15;
  v17 = (unsigned __int64 *)a1[2];
  if ( v17 == (unsigned __int64 *)a1[3] )
  {
    sub_2C83610(a1 + 1, v17, (__int64 *)v24);
    v15 = v24[0];
  }
  else
  {
    if ( v17 )
    {
      *v17 = v15;
      a1[2] += 8LL;
      return;
    }
    a1[2] = 8;
  }
  if ( v15 )
    j_j___libc_free_0(v15);
}
