// Function: sub_34BEAF0
// Address: 0x34beaf0
//
void __fastcall sub_34BEAF0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v5; // rbx
  __int64 v6; // r11
  _QWORD *v7; // rdx
  __int64 (*v8)(); // rax
  void (__fastcall *v9)(__int64, __int64, __int64, _QWORD, _BYTE **, _QWORD, __int64 *, _QWORD); // rax
  __int64 v10; // rsi
  __int64 v11; // rsi
  char v12; // al
  __int64 (*v13)(); // rax
  char v14; // al
  __int64 v17; // [rsp+10h] [rbp-120h]
  __int64 v18; // [rsp+18h] [rbp-118h]
  __int64 v19; // [rsp+28h] [rbp-108h] BYREF
  __int64 v20; // [rsp+30h] [rbp-100h] BYREF
  __int64 v21; // [rsp+38h] [rbp-F8h] BYREF
  unsigned __int64 *v22; // [rsp+40h] [rbp-F0h]
  __int64 v23; // [rsp+48h] [rbp-E8h]
  _BYTE *v24; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v25; // [rsp+58h] [rbp-D8h]
  _BYTE v26[208]; // [rsp+60h] [rbp-D0h] BYREF

  v5 = *(_QWORD *)(a1 + 32);
  v18 = *(_QWORD *)(a1 + 8);
  v19 = 0;
  v20 = 0;
  v24 = v26;
  v25 = 0x400000000LL;
  sub_2E32880(&v21, a1);
  v6 = a2;
  if ( !v21 )
  {
    v11 = *a4;
    v21 = v11;
    if ( v11 )
    {
      sub_B96E90((__int64)&v21, v11, 1);
      v6 = a2;
    }
  }
  v7 = *(_QWORD **)a3;
  if ( v18 != v5 + 320 )
  {
    v8 = (__int64 (*)())v7[43];
    if ( v8 != sub_2DB1AE0 )
    {
      v17 = v6;
      v12 = ((__int64 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, __int64))v8)(
              a3,
              a1,
              &v19,
              &v20,
              &v24,
              1);
      v7 = *(_QWORD **)a3;
      v6 = v17;
      if ( !v12 && v19 == v18 )
      {
        if ( (_DWORD)v25 )
        {
          if ( !v20 )
          {
            v13 = (__int64 (*)())v7[110];
            if ( v13 != sub_2DB1B20 )
            {
              v14 = ((__int64 (__fastcall *)(__int64, _BYTE **))v13)(a3, &v24);
              v6 = v17;
              if ( !v14 )
              {
                (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a3 + 360LL))(a3, a1, 0);
                (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 *))(*(_QWORD *)a3 + 368LL))(
                  a3,
                  a1,
                  v17,
                  0,
                  v24,
                  (unsigned int)v25,
                  &v21);
                v10 = v21;
                if ( !v21 )
                  goto LABEL_8;
LABEL_7:
                sub_B91220((__int64)&v21, v10);
                goto LABEL_8;
              }
              v7 = *(_QWORD **)a3;
            }
          }
        }
      }
    }
  }
  v9 = (void (__fastcall *)(__int64, __int64, __int64, _QWORD, _BYTE **, _QWORD, __int64 *, _QWORD))v7[46];
  v22 = (unsigned __int64 *)&v24;
  v23 = 0;
  v9(a3, a1, v6, 0, &v24, 0, &v21, 0);
  if ( v22 != (unsigned __int64 *)&v24 )
    _libc_free((unsigned __int64)v22);
  v10 = v21;
  if ( v21 )
    goto LABEL_7;
LABEL_8:
  if ( v24 != v26 )
    _libc_free((unsigned __int64)v24);
}
