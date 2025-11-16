// Function: sub_20D69B0
// Address: 0x20d69b0
//
void __fastcall sub_20D69B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r10
  __int64 v5; // r11
  _QWORD *v6; // rdx
  __int64 (*v7)(); // rax
  void (__fastcall *v8)(__int64, __int64, __int64, _QWORD, _BYTE **, _QWORD, __int64 *, _QWORD); // rax
  __int64 v9; // rsi
  char v10; // al
  __int64 (*v11)(); // rax
  char v12; // al
  __int64 v14; // [rsp+18h] [rbp-118h]
  __int64 v15; // [rsp+28h] [rbp-108h] BYREF
  __int64 v16; // [rsp+30h] [rbp-100h] BYREF
  __int64 v17; // [rsp+38h] [rbp-F8h] BYREF
  unsigned __int64 *v18; // [rsp+40h] [rbp-F0h]
  __int64 v19; // [rsp+48h] [rbp-E8h]
  _BYTE *v20; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v21; // [rsp+58h] [rbp-D8h]
  _BYTE v22[208]; // [rsp+60h] [rbp-D0h] BYREF

  v14 = *(_QWORD *)(a1 + 8);
  v3 = *(_QWORD *)(a1 + 56) + 320LL;
  v15 = 0;
  v16 = 0;
  v20 = v22;
  v21 = 0x400000000LL;
  sub_1DD6F40(&v17, a1);
  v4 = a3;
  v5 = a2;
  v6 = *(_QWORD **)a3;
  if ( v14 != v3 )
  {
    v7 = (__int64 (*)())v6[33];
    if ( v7 != sub_1D820E0 )
    {
      v10 = ((__int64 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, __int64))v7)(
              a3,
              a1,
              &v15,
              &v16,
              &v20,
              1);
      v4 = a3;
      v5 = a2;
      v6 = *(_QWORD **)a3;
      if ( !v10 && v15 == v14 )
      {
        if ( (_DWORD)v21 )
        {
          if ( !v16 )
          {
            v11 = (__int64 (*)())v6[78];
            if ( v11 != sub_1D918B0 )
            {
              v12 = ((__int64 (__fastcall *)(__int64, _BYTE **))v11)(a3, &v20);
              v4 = a3;
              v5 = a2;
              if ( !v12 )
              {
                (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a3 + 280LL))(a3, a1, 0);
                (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 *))(*(_QWORD *)a3 + 288LL))(
                  a3,
                  a1,
                  a2,
                  0,
                  v20,
                  (unsigned int)v21,
                  &v17);
                v9 = v17;
                if ( !v17 )
                  goto LABEL_7;
LABEL_6:
                sub_161E7C0((__int64)&v17, v9);
                goto LABEL_7;
              }
              v6 = *(_QWORD **)a3;
            }
          }
        }
      }
    }
  }
  v8 = (void (__fastcall *)(__int64, __int64, __int64, _QWORD, _BYTE **, _QWORD, __int64 *, _QWORD))v6[36];
  v18 = (unsigned __int64 *)&v20;
  v19 = 0;
  v8(v4, a1, v5, 0, &v20, 0, &v17, 0);
  if ( v18 != (unsigned __int64 *)&v20 )
    _libc_free((unsigned __int64)v18);
  v9 = v17;
  if ( v17 )
    goto LABEL_6;
LABEL_7:
  if ( v20 != v22 )
    _libc_free((unsigned __int64)v20);
}
