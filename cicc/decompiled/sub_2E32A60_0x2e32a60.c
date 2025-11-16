// Function: sub_2E32A60
// Address: 0x2e32a60
//
void __fastcall sub_2E32A60(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 (*v3)(void); // rax
  __int64 (*v4)(); // rax
  __int64 (*v5)(); // rax
  __int64 (*v6)(); // rax
  __int64 v7; // r9
  unsigned __int64 v8; // r8
  void (__fastcall *v9)(__int64, __int64, __int64, _QWORD, unsigned __int64, __int64); // rax
  __int64 v10; // [rsp+18h] [rbp-F8h] BYREF
  __int64 v11; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v12; // [rsp+28h] [rbp-E8h] BYREF
  _BYTE *v13; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v14; // [rsp+38h] [rbp-D8h]
  _BYTE v15[208]; // [rsp+40h] [rbp-D0h] BYREF

  v2 = 0;
  v3 = *(__int64 (**)(void))(**(_QWORD **)(*(_QWORD *)(a1 + 32) + 16LL) + 128LL);
  if ( v3 != sub_2DAC790 )
    v2 = v3();
  if ( *(_DWORD *)(a1 + 120) )
  {
    v10 = 0;
    v14 = 0x400000000LL;
    v11 = 0;
    v13 = v15;
    sub_2E32880(&v12, a1);
    v4 = *(__int64 (**)())(*(_QWORD *)v2 + 344LL);
    if ( v4 != sub_2DB1AE0 )
      ((void (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v4)(v2, a1, &v10, &v11, &v13, 0);
    if ( !(_DWORD)v14 )
    {
      if ( v10 )
      {
        if ( sub_2E322F0(a1, v10) )
          (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v2 + 360LL))(v2, a1, 0);
        goto LABEL_17;
      }
      if ( a2 && sub_2E322C0(a1, a2) && !*(_BYTE *)(a2 + 216) && !sub_2E322F0(a1, a2) )
LABEL_13:
        (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 *))(*(_QWORD *)v2 + 368LL))(
          v2,
          a1,
          a2,
          0,
          v13,
          (unsigned int)v14,
          &v12);
LABEL_17:
      if ( v12 )
        sub_B91220((__int64)&v12, v12);
      if ( v13 != v15 )
        _libc_free((unsigned __int64)v13);
      return;
    }
    if ( v11 )
    {
      if ( sub_2E322F0(a1, v10) )
      {
        v5 = *(__int64 (**)())(*(_QWORD *)v2 + 880LL);
        if ( v5 != sub_2DB1B20 && !((unsigned __int8 (__fastcall *)(__int64, _BYTE **))v5)(v2, &v13) )
        {
          (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v2 + 360LL))(v2, a1, 0);
          (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 *, _QWORD))(*(_QWORD *)v2 + 368LL))(
            v2,
            a1,
            v11,
            0,
            v13,
            (unsigned int)v14,
            &v12,
            0);
        }
        goto LABEL_17;
      }
      if ( !sub_2E322F0(a1, v11) )
        goto LABEL_17;
      (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v2 + 360LL))(v2, a1, 0);
      v8 = (unsigned __int64)v13;
      v7 = (unsigned int)v14;
      v9 = *(void (__fastcall **)(__int64, __int64, __int64, _QWORD, unsigned __int64, __int64))(*(_QWORD *)v2 + 368LL);
    }
    else
    {
      if ( v10 != a2 )
      {
        if ( !sub_2E322F0(a1, v10) )
        {
          if ( !sub_2E322F0(a1, a2) )
          {
            (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v2 + 360LL))(v2, a1, 0);
            (*(void (__fastcall **)(__int64, __int64, __int64, __int64, _BYTE *, _QWORD))(*(_QWORD *)v2 + 368LL))(
              v2,
              a1,
              v10,
              a2,
              v13,
              (unsigned int)v14);
          }
          goto LABEL_17;
        }
        v6 = *(__int64 (**)())(*(_QWORD *)v2 + 880LL);
        if ( v6 == sub_2DB1B20 || ((unsigned __int8 (__fastcall *)(__int64, _BYTE **))v6)(v2, &v13) )
        {
          LODWORD(v14) = 0;
          (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 *))(*(_QWORD *)v2 + 368LL))(
            v2,
            a1,
            a2,
            0,
            v13,
            0,
            &v12);
          goto LABEL_17;
        }
        (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v2 + 360LL))(v2, a1, 0);
        goto LABEL_13;
      }
      (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v2 + 360LL))(v2, a1, 0);
      if ( sub_2E322F0(a1, v10) )
        goto LABEL_17;
      LODWORD(v14) = 0;
      v7 = 0;
      v8 = (unsigned __int64)v13;
      v9 = *(void (__fastcall **)(__int64, __int64, __int64, _QWORD, unsigned __int64, __int64))(*(_QWORD *)v2 + 368LL);
    }
    v9(v2, a1, v10, 0, v8, v7);
    goto LABEL_17;
  }
}
