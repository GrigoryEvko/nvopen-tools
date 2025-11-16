// Function: sub_C0A1C0
// Address: 0xc0a1c0
//
unsigned __int64 __fastcall sub_C0A1C0(int *a1, _QWORD **a2)
{
  __int64 v2; // rbx
  __int64 v3; // r10
  __int64 v4; // r13
  char v5; // r12
  __int64 v6; // r14
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  int v9; // eax
  __int64 *v10; // r9
  __int64 v11; // rdi
  char v12; // al
  const void *v13; // rsi
  unsigned __int64 v14; // r12
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 *v19; // [rsp+0h] [rbp-C0h]
  __int64 v20; // [rsp+0h] [rbp-C0h]
  int i; // [rsp+1Ch] [rbp-A4h]
  __int64 v22; // [rsp+28h] [rbp-98h]
  _QWORD *v23; // [rsp+40h] [rbp-80h] BYREF
  __int64 v24; // [rsp+48h] [rbp-78h]
  _BYTE v25[112]; // [rsp+50h] [rbp-70h] BYREF

  LODWORD(v2) = 0;
  v23 = v25;
  v3 = (unsigned int)a1[4];
  v4 = *((_QWORD *)a1 + 1);
  v5 = *((_BYTE *)a1 + 4);
  v24 = 0x800000000LL;
  v3 *= 16;
  v6 = v4 + v3;
  v22 = *(_QWORD *)a1;
  for ( i = *a1; v6 != v4; v4 += 16 )
  {
    v9 = *(_DWORD *)(v4 + 4);
    if ( v9 == 10 )
    {
      v16 = (__int64 *)sub_BCB2A0(*a2);
      BYTE4(v22) = v5;
      LODWORD(v22) = i;
      v17 = sub_BCE1B0(v16, v22);
      v18 = (unsigned int)v24;
      if ( (unsigned __int64)(unsigned int)v24 + 1 > HIDWORD(v24) )
      {
        v20 = v17;
        sub_C8D5F0(&v23, v25, (unsigned int)v24 + 1LL, 8);
        v18 = (unsigned int)v24;
        v17 = v20;
      }
      v23[v18] = v17;
      LODWORD(v24) = v24 + 1;
      continue;
    }
    v2 = (unsigned int)(v2 + 1);
    v10 = (__int64 *)a2[2][v2];
    if ( v9 )
    {
      v7 = (unsigned int)v24;
      v8 = (unsigned int)v24 + 1LL;
      if ( v8 > HIDWORD(v24) )
        goto LABEL_9;
    }
    else
    {
      BYTE4(v22) = v5;
      LODWORD(v22) = i;
      v10 = (__int64 *)sub_BCE1B0(v10, v22);
      v7 = (unsigned int)v24;
      v8 = (unsigned int)v24 + 1LL;
      if ( v8 > HIDWORD(v24) )
      {
LABEL_9:
        v19 = v10;
        sub_C8D5F0(&v23, v25, v8, 8);
        v7 = (unsigned int)v24;
        v10 = v19;
      }
    }
    v23[v7] = v10;
    LODWORD(v24) = v24 + 1;
  }
  v11 = *a2[2];
  v12 = *(_BYTE *)(v11 + 8);
  if ( v12 != 7 )
  {
    BYTE4(v22) = v5;
    LODWORD(v22) = i;
    if ( v12 == 15 )
    {
      v11 = sub_E454C0(v11, v22);
    }
    else if ( v12 != 9 && (v5 || i != 1) )
    {
      v11 = sub_BCE1B0((__int64 *)v11, v22);
    }
  }
  v13 = v23;
  v14 = sub_BCF480((__int64 *)v11, v23, (unsigned int)v24, 0);
  if ( v23 != (_QWORD *)v25 )
    _libc_free(v23, v13);
  return v14;
}
