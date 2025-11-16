// Function: sub_CD61D0
// Address: 0xcd61d0
//
__int64 __fastcall sub_CD61D0(__int64 a1, __int64 *a2)
{
  int v2; // ebx
  __int64 v3; // rax
  unsigned __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rsi
  unsigned __int64 v8; // rax
  _DWORD *v9; // rbx
  int v10; // eax
  char v11; // al
  _BOOL8 v12; // rcx
  char v13; // al
  _BOOL8 v14; // rcx
  char v15; // al
  _BOOL8 v16; // rcx
  __int64 v19; // [rsp+28h] [rbp-68h]
  char v20; // [rsp+3Fh] [rbp-51h] BYREF
  __int64 v21; // [rsp+40h] [rbp-50h] BYREF
  __int64 v22; // [rsp+48h] [rbp-48h] BYREF
  __int64 v23; // [rsp+50h] [rbp-40h] BYREF
  _DWORD *v24; // [rsp+58h] [rbp-38h]

  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v2 = -1431655765 * ((a2[1] - *a2) >> 2);
  if ( v2 )
  {
    v3 = (unsigned int)(v2 - 1);
    v4 = 1;
    v5 = 0;
    v19 = v3 + 2;
    do
    {
      while ( 1 )
      {
        v6 = v5 + 12;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(
               a1,
               (unsigned int)(v4 - 1),
               &v21) )
        {
          break;
        }
        v5 += 12;
        if ( ++v4 == v19 )
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
      }
      v7 = *a2;
      v8 = 0xAAAAAAAAAAAAAAABLL * ((a2[1] - *a2) >> 2);
      if ( v8 <= v4 - 1 )
      {
        if ( v8 < v4 )
        {
          sub_CD5FF0((__int64)a2, v4 - v8);
          v7 = *a2;
        }
        else if ( v8 > v4 && a2[1] != v7 + v6 )
        {
          a2[1] = v7 + v6;
        }
      }
      v9 = (_DWORD *)(v7 + v5);
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
      v10 = *v9;
      v24 = v9;
      LODWORD(v23) = v10;
      v11 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
      v12 = 0;
      if ( v11 )
        v12 = (_DWORD)v23 == 0;
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "candNum",
             0,
             v12,
             &v20,
             &v22) )
      {
        sub_CCC2C0(a1, (unsigned int *)&v23);
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v22);
      }
      else if ( v20 )
      {
        LODWORD(v23) = 0;
      }
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
        *v24 = v23;
      LODWORD(v23) = v9[1];
      v24 = v9 + 1;
      v13 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
      v14 = 0;
      if ( v13 )
        v14 = (_DWORD)v23 == 0;
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "lineNo",
             0,
             v14,
             &v20,
             &v22) )
      {
        sub_CCC2C0(a1, (unsigned int *)&v23);
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v22);
      }
      else if ( v20 )
      {
        LODWORD(v23) = 0;
      }
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
        *v24 = v23;
      v15 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
      v16 = 0;
      if ( v15 )
        v16 = *((float *)v9 + 2) == 0.0;
      if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, __int64 *, __int64 *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "weight",
             0,
             v16,
             &v22,
             &v23) )
      {
        sub_CCC830(a1, v9 + 2);
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v23);
      }
      else if ( (_BYTE)v22 )
      {
        v9[2] = 0;
      }
      v5 = v6;
      ++v4;
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL))(a1, v21);
    }
    while ( v4 != v19 );
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
}
