// Function: sub_3117B40
// Address: 0x3117b40
//
__int64 __fastcall sub_3117B40(__int64 a1, __int64 a2)
{
  bool v4; // zf
  __int64 v5; // rax
  __int64 v6; // r15
  unsigned __int64 v7; // rcx
  char *v8; // r8
  size_t v9; // r9
  _QWORD *v10; // rax
  unsigned __int64 v12; // rax
  _BYTE *v13; // rsi
  __int64 v14; // rax
  _QWORD *v15; // rdi
  unsigned __int64 v16; // r13
  unsigned __int64 v17; // r14
  _BYTE *v18; // rsi
  unsigned __int64 v19; // rdx
  char *src; // [rsp+8h] [rbp-98h]
  __int64 i; // [rsp+18h] [rbp-88h]
  _BYTE *v22; // [rsp+28h] [rbp-78h] BYREF
  __int64 v23; // [rsp+30h] [rbp-70h] BYREF
  char v24; // [rsp+44h] [rbp-5Ch] BYREF
  _BYTE v25[11]; // [rsp+45h] [rbp-5Bh] BYREF
  _QWORD *v26; // [rsp+50h] [rbp-50h] BYREF
  _QWORD *v27; // [rsp+58h] [rbp-48h]
  _QWORD v28[8]; // [rsp+60h] [rbp-40h] BYREF

  v4 = (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) == 0;
  v5 = *(_QWORD *)a1;
  if ( v4 )
  {
    (*(void (**)(void))(v5 + 104))();
    (*(void (__fastcall **)(_QWORD **, __int64))(*(_QWORD *)a1 + 136LL))(&v26, a1);
    v16 = (unsigned __int64)v26;
    v17 = (unsigned __int64)v27;
    if ( v26 != v27 )
    {
      do
      {
        v18 = *(_BYTE **)v16;
        v19 = *(_QWORD *)(v16 + 8);
        v16 += 16LL;
        sub_3117840(a1, v18, v19, a2);
      }
      while ( v17 != v16 );
      v17 = (unsigned __int64)v26;
    }
    if ( v17 )
      j_j___libc_free_0(v17);
  }
  else
  {
    (*(void (**)(void))(v5 + 104))();
    v6 = *(_QWORD *)(a2 + 24);
    for ( i = a2 + 8; v6 != i; v6 = sub_220EEE0(v6) )
    {
      v7 = *(unsigned int *)(v6 + 32);
      if ( *(_DWORD *)(v6 + 32) )
      {
        v8 = v25;
        do
        {
          *--v8 = v7 % 0xA + 48;
          v12 = v7;
          v7 /= 0xAu;
        }
        while ( v12 > 9 );
        v13 = (_BYTE *)(v25 - v8);
        v26 = v28;
        v22 = (_BYTE *)(v25 - v8);
        v9 = v25 - v8;
        if ( (unsigned __int64)(v25 - v8) > 0xF )
        {
          src = v8;
          v14 = sub_22409D0((__int64)&v26, (unsigned __int64 *)&v22, 0);
          v8 = src;
          v9 = (size_t)v13;
          v26 = (_QWORD *)v14;
          v15 = (_QWORD *)v14;
          v28[0] = v22;
LABEL_16:
          memcpy(v15, v8, v9);
          v9 = (size_t)v22;
          v10 = v26;
          goto LABEL_6;
        }
        if ( v13 != (_BYTE *)1 )
        {
          if ( !v13 )
          {
            v10 = v28;
            goto LABEL_6;
          }
          v15 = v28;
          goto LABEL_16;
        }
      }
      else
      {
        v24 = 48;
        v8 = &v24;
        v26 = v28;
      }
      v9 = 1;
      LOBYTE(v28[0]) = *v8;
      v10 = v28;
LABEL_6:
      v27 = (_QWORD *)v9;
      *((_BYTE *)v10 + v9) = 0;
      if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD *, __int64, _QWORD, _BYTE **, __int64 *))(*(_QWORD *)a1 + 120LL))(
             a1,
             v26,
             1,
             0,
             &v22,
             &v23) )
      {
        sub_3117620(a1, v6 + 40);
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v23);
      }
      if ( v26 != v28 )
        j_j___libc_free_0((unsigned __int64)v26);
    }
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
}
