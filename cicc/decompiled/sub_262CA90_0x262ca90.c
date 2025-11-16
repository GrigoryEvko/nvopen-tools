// Function: sub_262CA90
// Address: 0x262ca90
//
__int64 __fastcall sub_262CA90(__int64 a1, _QWORD *a2)
{
  bool v3; // zf
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v7; // rdx
  _QWORD *v8; // rsi
  _BYTE *v9; // rsi
  __int64 v10; // r12
  unsigned __int64 v11; // r13
  _BYTE *v12; // rsi
  __int64 v13; // rdx
  _QWORD *v14; // [rsp+18h] [rbp-68h]
  char v15; // [rsp+27h] [rbp-59h] BYREF
  __int64 v16; // [rsp+28h] [rbp-58h] BYREF
  _QWORD *v17; // [rsp+30h] [rbp-50h] BYREF
  _QWORD *v18; // [rsp+38h] [rbp-48h]
  _BYTE v19[64]; // [rsp+40h] [rbp-40h] BYREF

  v3 = (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) == 0;
  v4 = *(_QWORD *)a1;
  if ( v3 )
  {
    (*(void (**)(void))(v4 + 104))();
    (*(void (__fastcall **)(_QWORD **, __int64))(*(_QWORD *)a1 + 136LL))(&v17, a1);
    v10 = (__int64)v17;
    v11 = (unsigned __int64)v18;
    if ( v17 != v18 )
    {
      do
      {
        v12 = *(_BYTE **)v10;
        v13 = *(_QWORD *)(v10 + 8);
        v10 += 16;
        sub_262C710(a1, v12, v13, a2);
      }
      while ( v11 != v10 );
      v11 = (unsigned __int64)v17;
    }
    if ( v11 )
      j_j___libc_free_0(v11);
  }
  else
  {
    (*(void (**)(void))(v4 + 104))();
    v5 = a2[3];
    v14 = a2 + 1;
    if ( (_QWORD *)v5 != a2 + 1 )
    {
      do
      {
        v9 = *(_BYTE **)(v5 + 40);
        if ( v9 )
        {
          v7 = *(_QWORD *)(v5 + 48);
          v17 = v19;
          sub_2619AF0((__int64 *)&v17, v9, (__int64)&v9[v7]);
          v8 = v17;
        }
        else
        {
          v8 = v19;
          v17 = v19;
          v18 = 0;
          v19[0] = 0;
        }
        if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD *, __int64, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
               a1,
               v8,
               1,
               0,
               &v15,
               &v16) )
        {
          sub_262C1A0(a1, v5 + 56);
          (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v16);
        }
        if ( v17 != (_QWORD *)v19 )
          j_j___libc_free_0((unsigned __int64)v17);
        v5 = sub_220EEE0(v5);
      }
      while ( v14 != (_QWORD *)v5 );
    }
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
}
