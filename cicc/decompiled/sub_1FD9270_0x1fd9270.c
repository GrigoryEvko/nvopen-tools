// Function: sub_1FD9270
// Address: 0x1fd9270
//
unsigned __int64 __fastcall sub_1FD9270(_QWORD *a1, __int64 *a2)
{
  unsigned int v2; // eax
  unsigned int v4; // ebx
  unsigned int v5; // eax
  unsigned int v6; // r15d
  int v7; // eax
  unsigned int v8; // r14d
  __int64 v9; // rdx
  unsigned int v10; // r13d
  unsigned int v11; // eax
  __int64 (*v12)(); // rax
  unsigned __int8 v13; // r14
  unsigned int v14; // r13d
  unsigned int v15; // eax
  _BOOL8 v16; // r9
  __int64 v17; // r8
  __int64 v18; // rcx
  __int64 v19; // rsi
  int v20; // eax
  __int64 v21; // rdx
  bool v22; // [rsp+Fh] [rbp-51h]
  int v23; // [rsp+10h] [rbp-50h] BYREF
  __int64 v24; // [rsp+18h] [rbp-48h]
  _BYTE v25[8]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v26; // [rsp+28h] [rbp-38h]

  v2 = sub_1FD8F60(a1, (__int64)a2);
  if ( !v2 )
    return 0;
  v4 = v2;
  v22 = sub_1FD4DC0((__int64)a1, (__int64)a2);
  v5 = 8 * sub_15A9520(a1[12], 0);
  if ( v5 == 32 )
  {
    v6 = 5;
    goto LABEL_7;
  }
  if ( v5 > 0x20 )
  {
    if ( v5 == 64 )
    {
      v6 = 6;
      goto LABEL_7;
    }
    if ( v5 == 128 )
    {
      v6 = 7;
      goto LABEL_7;
    }
LABEL_36:
    LOBYTE(v20) = sub_1F59570(*a2);
    v25[0] = 0;
    v23 = v20;
    v8 = v20;
    v24 = v21;
    v26 = 0;
    if ( (_BYTE)v20 )
    {
      v10 = sub_1FD3510(v20);
    }
    else
    {
      if ( !v21 )
        goto LABEL_32;
      v10 = sub_1F58D40((__int64)&v23);
    }
    v6 = 0;
    v11 = sub_1F58D40((__int64)v25);
    goto LABEL_11;
  }
  if ( v5 == 8 )
  {
    v6 = 3;
    goto LABEL_7;
  }
  v6 = 4;
  if ( v5 != 16 )
    goto LABEL_36;
LABEL_7:
  LOBYTE(v7) = sub_1F59570(*a2);
  v25[0] = v6;
  v23 = v7;
  v8 = v7;
  v24 = v9;
  v26 = 0;
  if ( (_BYTE)v7 == (_BYTE)v6 )
    goto LABEL_20;
  if ( (_BYTE)v7 )
    v10 = sub_1FD3510(v7);
  else
    v10 = sub_1F58D40((__int64)&v23);
  v11 = sub_1FD3510(v6);
LABEL_11:
  if ( v10 >= v11 )
  {
    v25[0] = v6;
    v26 = 0;
    if ( (_BYTE)v6 != (_BYTE)v8 )
    {
      if ( (_BYTE)v8 )
        v14 = sub_1FD3510(v8);
      else
        v14 = sub_1F58D40((__int64)&v23);
      if ( (_BYTE)v6 )
      {
        v15 = sub_1FD3510(v6);
        goto LABEL_27;
      }
LABEL_34:
      v6 = 0;
      v15 = sub_1F58D40((__int64)v25);
LABEL_27:
      if ( v15 < v14 )
      {
        v12 = *(__int64 (**)())(*a1 + 64LL);
        if ( v12 == sub_1FD34C0 )
          goto LABEL_13;
        v16 = v22;
        v17 = v4;
        v18 = 145;
LABEL_30:
        v19 = v8;
        v13 = 1;
        v4 = ((__int64 (__fastcall *)(_QWORD *, __int64, _QWORD, __int64, __int64, _BOOL8))v12)(
               a1,
               v19,
               v6,
               v18,
               v17,
               v16);
        return ((unsigned __int64)v13 << 32) | v4;
      }
LABEL_20:
      v13 = v22;
      return ((unsigned __int64)v13 << 32) | v4;
    }
    if ( (_BYTE)v6 )
      goto LABEL_20;
LABEL_32:
    if ( !v24 )
      goto LABEL_20;
    v8 = 0;
    v14 = sub_1F58D40((__int64)&v23);
    goto LABEL_34;
  }
  v12 = *(__int64 (**)())(*a1 + 64LL);
  if ( v12 != sub_1FD34C0 )
  {
    v16 = v22;
    v17 = v4;
    v18 = 142;
    goto LABEL_30;
  }
LABEL_13:
  v13 = 1;
  v4 = 0;
  return ((unsigned __int64)v13 << 32) | v4;
}
