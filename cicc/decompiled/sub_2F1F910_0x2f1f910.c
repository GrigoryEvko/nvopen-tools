// Function: sub_2F1F910
// Address: 0x2f1f910
//
__int64 __fastcall sub_2F1F910(__int64 a1, unsigned __int64 *a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v4; // rax
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rbx
  char i; // al
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rax
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // [rsp+8h] [rbp-78h]
  unsigned int *v17; // [rsp+18h] [rbp-68h]
  unsigned __int64 v18; // [rsp+18h] [rbp-68h]
  __int64 v19; // [rsp+28h] [rbp-58h]
  char v20; // [rsp+3Fh] [rbp-41h] BYREF
  __int64 v21; // [rsp+40h] [rbp-40h] BYREF
  _QWORD v22[7]; // [rsp+48h] [rbp-38h] BYREF

  v2 = a1;
  LODWORD(v3) = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v3 = (__int64)(a2[1] - *a2) >> 6;
  if ( (_DWORD)v3 )
  {
    v4 = (unsigned int)(v3 - 1);
    v5 = 1;
    v6 = 0;
    v19 = v4 + 2;
    for ( i = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(a1, 0, &v21);
          ;
          i = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(a1, (unsigned int)v6, &v21) )
    {
      if ( i )
      {
        v9 = *a2;
        v10 = (__int64)(a2[1] - *a2) >> 6;
        if ( v10 <= v6 )
        {
          if ( v10 < v5 )
          {
            sub_2F1F6A0(a2, v5 - v10);
            v9 = *a2;
          }
          else if ( v10 > v5 )
          {
            v18 = v9 + (v5 << 6);
            if ( a2[1] != v18 )
            {
              v15 = v6;
              v12 = a2[1];
              v13 = v9 + (v5 << 6);
              do
              {
                v14 = *(_QWORD *)(v13 + 8);
                if ( v14 != v13 + 24 )
                  j_j___libc_free_0(v14);
                v13 += 64LL;
              }
              while ( v12 != v13 );
              v6 = v15;
              a2[1] = v18;
              v9 = *a2;
            }
          }
        }
        v17 = (unsigned int *)(v9 + (v6 << 6));
        (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
        if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
               a1,
               "bb",
               1,
               0,
               &v20,
               v22) )
        {
          sub_2F07DB0(a1, v17);
          (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v22[0]);
        }
        if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
               a1,
               "offset",
               1,
               0,
               &v20,
               v22) )
        {
          sub_2F07DB0(a1, v17 + 1);
          (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v22[0]);
        }
        if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
               a1,
               "callee",
               1,
               0,
               &v20,
               v22) )
        {
          sub_2F0E9C0(a1, (__int64)(v17 + 2));
          (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v22[0]);
        }
        if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
               a1,
               "flags",
               1,
               0,
               &v20,
               v22) )
        {
          sub_2F07DB0(a1, v17 + 14);
          (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v22[0]);
        }
        ++v6;
        ++v5;
        (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL))(a1, v21);
        if ( v19 == v5 )
        {
LABEL_18:
          v2 = a1;
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 48LL))(v2);
        }
      }
      else
      {
        ++v6;
        if ( v19 == ++v5 )
          goto LABEL_18;
      }
    }
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 48LL))(v2);
}
