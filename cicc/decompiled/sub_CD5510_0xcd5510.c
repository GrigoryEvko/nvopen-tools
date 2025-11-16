// Function: sub_CD5510
// Address: 0xcd5510
//
__int64 __fastcall sub_CD5510(__int64 a1, __int64 *a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rbx
  char i; // al
  __int64 v7; // r12
  unsigned __int64 v8; // rax
  _DWORD *v9; // r12
  int v10; // eax
  char v11; // al
  _BOOL8 v12; // rcx
  char v13; // al
  _BOOL8 v14; // rcx
  char v15; // al
  _BOOL8 v16; // rcx
  char v17; // al
  _BOOL8 v18; // rcx
  __int64 v20; // rax
  __int64 v21; // [rsp+28h] [rbp-68h]
  char v22; // [rsp+3Fh] [rbp-51h] BYREF
  __int64 v23; // [rsp+40h] [rbp-50h] BYREF
  __int64 v24; // [rsp+48h] [rbp-48h] BYREF
  __int64 v25; // [rsp+50h] [rbp-40h] BYREF
  _DWORD *v26; // [rsp+58h] [rbp-38h]

  LODWORD(v2) = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v2 = (a2[1] - *a2) >> 4;
  if ( (_DWORD)v2 )
  {
    v3 = (unsigned int)(v2 - 1);
    v4 = 1;
    v5 = 0;
    v21 = v3 + 2;
    for ( i = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(a1, 0, &v23);
          ;
          i = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(a1, (unsigned int)v5, &v23) )
    {
      if ( i )
      {
        v7 = *a2;
        v8 = (a2[1] - *a2) >> 4;
        if ( v8 <= v5 )
        {
          if ( v8 < v4 )
          {
            sub_CD5330((__int64)a2, v4 - v8);
            v7 = *a2;
          }
          else if ( v8 > v4 )
          {
            v20 = v7 + 16 * v4;
            if ( a2[1] != v20 )
              a2[1] = v20;
          }
        }
        v9 = (_DWORD *)(16 * v5 + v7);
        (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
        v10 = *v9;
        v26 = v9;
        LODWORD(v25) = v10;
        v11 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
        v12 = 0;
        if ( v11 )
          v12 = (_DWORD)v25 == 0;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
               a1,
               "candNum",
               0,
               v12,
               &v22,
               &v24) )
        {
          sub_CCC2C0(a1, (unsigned int *)&v25);
          (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v24);
        }
        else if ( v22 )
        {
          LODWORD(v25) = 0;
        }
        if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
          *v26 = v25;
        LODWORD(v25) = v9[1];
        v26 = v9 + 1;
        v13 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
        v14 = 0;
        if ( v13 )
          v14 = (_DWORD)v25 == 0;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
               a1,
               "lineNo",
               0,
               v14,
               &v22,
               &v24) )
        {
          sub_CCC2C0(a1, (unsigned int *)&v25);
          (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v24);
        }
        else if ( v22 )
        {
          LODWORD(v25) = 0;
        }
        if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
          *v26 = v25;
        v15 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
        v16 = 0;
        if ( v15 )
          v16 = *((float *)v9 + 2) == 0.0;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, __int64 *, __int64 *))(*(_QWORD *)a1 + 120LL))(
               a1,
               "weight",
               0,
               v16,
               &v24,
               &v25) )
        {
          sub_CCC830(a1, v9 + 2);
          (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v25);
        }
        else if ( (_BYTE)v24 )
        {
          v9[2] = 0;
        }
        v17 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
        v18 = 0;
        if ( v17 )
          v18 = *((float *)v9 + 3) == 0.0;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int64 *, __int64 *))(*(_QWORD *)a1 + 120LL))(
               a1,
               "pZero",
               0,
               v18,
               &v24,
               &v25) )
        {
          sub_CCC830(a1, v9 + 3);
          (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v25);
        }
        else if ( (_BYTE)v24 )
        {
          v9[3] = 0;
        }
        ++v5;
        ++v4;
        (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL))(a1, v23);
        if ( v4 == v21 )
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
      }
      else
      {
        ++v5;
        if ( ++v4 == v21 )
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
      }
    }
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
}
