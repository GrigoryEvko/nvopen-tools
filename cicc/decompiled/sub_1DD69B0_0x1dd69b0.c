// Function: sub_1DD69B0
// Address: 0x1dd69b0
//
__int64 __fastcall sub_1DD69B0(__int64 *a1)
{
  __int64 v1; // r12
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 (*v5)(); // rax
  __int64 v6; // r14
  __int64 (*v7)(); // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdi
  __int64 v10; // rdx
  __int16 v11; // ax
  unsigned __int64 v12; // rdx
  __int64 (*v13)(); // rdx
  unsigned __int64 v14; // rsi
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v18; // [rsp+0h] [rbp-F0h] BYREF
  __int64 v19; // [rsp+8h] [rbp-E8h] BYREF
  _BYTE *v20; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v21; // [rsp+18h] [rbp-D8h]
  _BYTE v22[208]; // [rsp+20h] [rbp-D0h] BYREF

  v1 = a1[1];
  if ( v1 == a1[7] + 320 || !sub_1DD6970((__int64)a1, a1[1]) )
    return 0;
  v18 = 0;
  v21 = 0x400000000LL;
  v3 = a1[7];
  v19 = 0;
  v4 = *(_QWORD *)(v3 + 16);
  v20 = v22;
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 40LL);
  if ( v5 == sub_1D00B00 )
    BUG();
  v6 = v5();
  v7 = *(__int64 (**)())(*(_QWORD *)v6 + 264LL);
  if ( v7 == sub_1D820E0
    || ((unsigned __int8 (__fastcall *)(__int64, __int64 *, __int64 *, __int64 *, _BYTE **, _QWORD))v7)(
         v6,
         a1,
         &v18,
         &v19,
         &v20,
         0) )
  {
    v8 = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
    v9 = v8;
    if ( (__int64 *)v8 == a1 + 3 )
      goto LABEL_22;
    if ( !v8 )
      BUG();
    v10 = *(_QWORD *)v8;
    v11 = *(_WORD *)(v8 + 46);
    if ( (v10 & 4) != 0 )
    {
      if ( (v11 & 4) != 0 )
        goto LABEL_27;
    }
    else if ( (v11 & 4) != 0 )
    {
      while ( 1 )
      {
        v12 = v10 & 0xFFFFFFFFFFFFFFF8LL;
        v11 = *(_WORD *)(v12 + 46);
        v9 = v12;
        if ( (v11 & 4) == 0 )
          break;
        v10 = *(_QWORD *)v12;
      }
    }
    if ( (v11 & 8) != 0 )
    {
      if ( !(unsigned __int8)sub_1E15D00(v9, 32, 1) )
        goto LABEL_22;
      v13 = *(__int64 (**)())(*(_QWORD *)v6 + 656LL);
      v14 = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v14 )
        BUG();
LABEL_15:
      v15 = *(_QWORD *)v14;
      if ( (*(_QWORD *)v14 & 4) == 0 && (*(_BYTE *)(v14 + 46) & 4) != 0 )
      {
        while ( 1 )
        {
          v16 = v15 & 0xFFFFFFFFFFFFFFF8LL;
          v14 = v16;
          if ( (*(_BYTE *)(v16 + 46) & 4) == 0 )
            break;
          v15 = *(_QWORD *)v16;
        }
      }
      if ( v13 != sub_1D918C0 && ((unsigned __int8 (__fastcall *)(__int64, unsigned __int64))v13)(v6, v14) )
        goto LABEL_22;
      goto LABEL_35;
    }
LABEL_27:
    if ( (*(_BYTE *)(*(_QWORD *)(v9 + 16) + 8LL) & 0x20) == 0 )
      goto LABEL_22;
    v14 = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
    v13 = *(__int64 (**)())(*(_QWORD *)v6 + 656LL);
    goto LABEL_15;
  }
  if ( v18 && v18 != v1 && v19 != v1 && (!(_DWORD)v21 || v19) )
LABEL_35:
    v1 = 0;
LABEL_22:
  if ( v20 != v22 )
    _libc_free((unsigned __int64)v20);
  return v1;
}
