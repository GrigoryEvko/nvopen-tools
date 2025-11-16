// Function: sub_1DD7120
// Address: 0x1dd7120
//
void __fastcall sub_1DD7120(__int64 *a1)
{
  __int64 v1; // r14
  __int64 (*v2)(void); // rax
  __int64 (*v3)(); // rax
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rsi
  __int64 v7; // r10
  __int64 (*v8)(); // rax
  __int64 *v9; // rax
  __int64 v10; // rcx
  bool v11; // al
  __int64 v12; // r10
  __int64 (*v13)(); // rax
  bool v14; // zf
  __int64 v15; // rax
  char v16; // al
  __int64 v17; // [rsp+8h] [rbp-108h]
  __int64 v18; // [rsp+18h] [rbp-F8h] BYREF
  __int64 v19; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v20; // [rsp+28h] [rbp-E8h] BYREF
  _BYTE *v21; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v22; // [rsp+38h] [rbp-D8h]
  _BYTE v23[208]; // [rsp+40h] [rbp-D0h] BYREF

  v1 = 0;
  v2 = *(__int64 (**)(void))(**(_QWORD **)(a1[7] + 16) + 40LL);
  if ( v2 != sub_1D00B00 )
    v1 = v2();
  if ( a1[11] != a1[12] )
  {
    v22 = 0x400000000LL;
    v18 = 0;
    v19 = 0;
    v21 = v23;
    sub_1DD6F40(&v20, (__int64)a1);
    v3 = *(__int64 (**)())(*(_QWORD *)v1 + 264LL);
    if ( v3 != sub_1D820E0 )
      ((void (__fastcall *)(__int64, __int64 *, __int64 *, __int64 *, _BYTE **, _QWORD))v3)(v1, a1, &v18, &v19, &v21, 0);
    if ( (_DWORD)v22 )
    {
      v7 = v19;
      if ( v19 )
      {
        if ( sub_1DD69A0((__int64)a1, v18) )
        {
          v8 = *(__int64 (**)())(*(_QWORD *)v1 + 624LL);
          if ( v8 != sub_1D918B0 && !((unsigned __int8 (__fastcall *)(__int64, _BYTE **))v8)(v1, &v21) )
          {
            (*(void (__fastcall **)(__int64, __int64 *, _QWORD))(*(_QWORD *)v1 + 280LL))(v1, a1, 0);
            (*(void (__fastcall **)(__int64, __int64 *, __int64, _QWORD, _BYTE *, _QWORD, __int64 *, _QWORD))(*(_QWORD *)v1 + 288LL))(
              v1,
              a1,
              v19,
              0,
              v21,
              (unsigned int)v22,
              &v20,
              0);
          }
          goto LABEL_15;
        }
        if ( sub_1DD69A0((__int64)a1, v19) )
        {
          (*(void (__fastcall **)(__int64, __int64 *, _QWORD))(*(_QWORD *)v1 + 280LL))(v1, a1, 0);
          goto LABEL_14;
        }
LABEL_15:
        if ( v20 )
          sub_161E7C0((__int64)&v20, v20);
        if ( v21 != v23 )
          _libc_free((unsigned __int64)v21);
        return;
      }
      v9 = (__int64 *)a1[11];
      v10 = a1[12];
      if ( (__int64 *)v10 != v9 )
      {
        do
        {
          if ( !*(_BYTE *)(*v9 + 180) && *v9 != v18 )
            v7 = *v9;
          ++v9;
        }
        while ( v9 != (__int64 *)v10 );
        if ( v7 )
        {
          v17 = v7;
          v11 = sub_1DD69A0((__int64)a1, v18);
          v12 = v17;
          if ( v11 )
          {
            v13 = *(__int64 (**)())(*(_QWORD *)v1 + 624LL);
            if ( v13 == sub_1D918B0
              || (v16 = ((__int64 (__fastcall *)(__int64, _BYTE **))v13)(v1, &v21), v12 = v17, v16) )
            {
              LODWORD(v22) = 0;
              (*(void (__fastcall **)(__int64, __int64 *, __int64, _QWORD, _BYTE *, _QWORD, __int64 *))(*(_QWORD *)v1 + 288LL))(
                v1,
                a1,
                v12,
                0,
                v21,
                0,
                &v20);
            }
            else
            {
              (*(void (__fastcall **)(__int64, __int64 *, _QWORD))(*(_QWORD *)v1 + 280LL))(v1, a1, 0);
              (*(void (__fastcall **)(__int64, __int64 *, __int64, _QWORD, _BYTE *, _QWORD, __int64 *))(*(_QWORD *)v1 + 288LL))(
                v1,
                a1,
                v17,
                0,
                v21,
                (unsigned int)v22,
                &v20);
            }
          }
          else if ( !sub_1DD69A0((__int64)a1, v17) )
          {
            (*(void (__fastcall **)(__int64, __int64 *, _QWORD))(*(_QWORD *)v1 + 280LL))(v1, a1, 0);
            (*(void (__fastcall **)(__int64, __int64 *, __int64, __int64, _BYTE *, _QWORD))(*(_QWORD *)v1 + 288LL))(
              v1,
              a1,
              v18,
              v17,
              v21,
              (unsigned int)v22);
          }
          goto LABEL_15;
        }
      }
      v14 = !sub_1DD6C00(a1);
      v15 = *(_QWORD *)v1;
      if ( v14 )
      {
        (*(void (__fastcall **)(__int64, __int64 *, _QWORD))(v15 + 280))(v1, a1, 0);
        LODWORD(v22) = 0;
        (*(void (__fastcall **)(__int64, __int64 *, __int64, _QWORD, _BYTE *, _QWORD))(*(_QWORD *)v1 + 288LL))(
          v1,
          a1,
          v18,
          0,
          v21,
          0);
        goto LABEL_15;
      }
      (*(void (__fastcall **)(__int64, __int64 *, _QWORD))(v15 + 280))(v1, a1, 0);
      v6 = v18;
    }
    else
    {
      if ( v18 )
      {
        if ( sub_1DD69A0((__int64)a1, v18) )
          (*(void (__fastcall **)(__int64, __int64 *, _QWORD))(*(_QWORD *)v1 + 280LL))(v1, a1, 0);
        goto LABEL_15;
      }
      v4 = a1[11];
      v5 = a1[12];
      if ( v4 == v5 )
        goto LABEL_15;
      do
      {
        if ( !*(_BYTE *)(*(_QWORD *)v4 + 180LL) )
          v18 = *(_QWORD *)v4;
        v4 += 8;
      }
      while ( v5 != v4 );
      v6 = v18;
      if ( !v18 )
        goto LABEL_15;
    }
    if ( !sub_1DD69A0((__int64)a1, v6) )
    {
LABEL_14:
      (*(void (__fastcall **)(__int64, __int64 *, __int64, _QWORD, _BYTE *, _QWORD))(*(_QWORD *)v1 + 288LL))(
        v1,
        a1,
        v18,
        0,
        v21,
        (unsigned int)v22);
      goto LABEL_15;
    }
    goto LABEL_15;
  }
}
