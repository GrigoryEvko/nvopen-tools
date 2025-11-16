// Function: sub_10BAE90
// Address: 0x10bae90
//
__int64 __fastcall sub_10BAE90(_QWORD *a1, _BYTE *a2)
{
  __int64 v4; // rsi
  unsigned int v5; // eax
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rdx
  unsigned int v8; // r13d
  bool v9; // zf
  __int64 v10; // rdi
  unsigned int v11; // r15d
  const void **v12; // r14
  _BYTE *v14; // r15
  unsigned int v15; // edx
  int v16; // eax
  __int64 v17; // rdx
  _BYTE *v18; // rax
  int v19; // eax
  int v20; // eax
  const void *v21; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v22; // [rsp+8h] [rbp-58h]
  const void *v23; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v24; // [rsp+18h] [rbp-48h]
  const void *v25; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v26; // [rsp+28h] [rbp-38h]

  v4 = a1[1];
  v5 = *(_DWORD *)(v4 + 8);
  v22 = v5;
  if ( v5 > 0x40 )
  {
    sub_C43780((__int64)&v21, (const void **)v4);
    v5 = v22;
    if ( v22 > 0x40 )
    {
      sub_C43D10((__int64)&v21);
      goto LABEL_6;
    }
    v6 = (unsigned __int64)v21;
  }
  else
  {
    v6 = *(_QWORD *)v4;
  }
  v7 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v5) & ~v6;
  if ( !v5 )
    v7 = 0;
  v21 = (const void *)v7;
LABEL_6:
  sub_C46250((__int64)&v21);
  v8 = v22;
  v22 = 0;
  v9 = *a2 == 42;
  v24 = v8;
  v23 = v21;
  if ( !v9 || *a1 != *((_QWORD *)a2 - 8) )
    goto LABEL_7;
  v14 = (_BYTE *)*((_QWORD *)a2 - 4);
  if ( !v14 )
    BUG();
  if ( *v14 != 17 )
  {
    v17 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v14 + 1) + 8LL) - 17;
    if ( (unsigned int)v17 > 1 )
      goto LABEL_7;
    if ( *v14 > 0x15u )
      goto LABEL_7;
    v18 = sub_AD7630(*((_QWORD *)a2 - 4), 1, v17);
    v8 = v24;
    v14 = v18;
    if ( !v18 || *v18 != 17 )
      goto LABEL_7;
  }
  v15 = *((_DWORD *)v14 + 8);
  v12 = (const void **)(v14 + 24);
  if ( v8 != v15 )
  {
    if ( v8 >= v15 )
    {
      sub_C449B0((__int64)&v25, v12, v8);
      if ( v26 <= 0x40 )
      {
        LOBYTE(v12) = v25 == v23;
        goto LABEL_30;
      }
      LOBYTE(v20) = sub_C43C50((__int64)&v25, &v23);
      LODWORD(v12) = v20;
    }
    else
    {
      sub_C449B0((__int64)&v25, &v23, v15);
      if ( *((_DWORD *)v14 + 8) <= 0x40u )
      {
        LOBYTE(v12) = *((_QWORD *)v14 + 3) == (_QWORD)v25;
      }
      else
      {
        LOBYTE(v16) = sub_C43C50((__int64)(v14 + 24), &v25);
        LODWORD(v12) = v16;
      }
      if ( v26 <= 0x40 )
        goto LABEL_30;
    }
    if ( v25 )
      j_j___libc_free_0_0(v25);
LABEL_30:
    v8 = v24;
    if ( (_BYTE)v12 )
      goto LABEL_9;
    goto LABEL_7;
  }
  if ( v8 <= 0x40 )
  {
    LODWORD(v12) = 1;
    if ( *((const void **)v14 + 3) == v23 )
      goto LABEL_12;
    goto LABEL_7;
  }
  LOBYTE(v19) = sub_C43C50((__int64)(v14 + 24), &v23);
  LODWORD(v12) = v19;
  if ( !(_BYTE)v19 )
  {
LABEL_7:
    v10 = a1[1];
    v11 = *(_DWORD *)(v10 + 8);
    if ( v11 <= 0x40 )
    {
      LODWORD(v12) = 0;
      if ( *(_QWORD *)v10 )
        goto LABEL_9;
    }
    else
    {
      LODWORD(v12) = 0;
      if ( v11 != (unsigned int)sub_C444A0(v10) )
        goto LABEL_9;
    }
    LOBYTE(v12) = *a1 == (_QWORD)a2;
LABEL_9:
    if ( v8 <= 0x40 )
      goto LABEL_12;
  }
  if ( v23 )
    j_j___libc_free_0_0(v23);
LABEL_12:
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  return (unsigned int)v12;
}
