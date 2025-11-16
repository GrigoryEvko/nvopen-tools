// Function: sub_B356A0
// Address: 0xb356a0
//
__int64 __fastcall sub_B356A0(unsigned int **a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, unsigned int *a6)
{
  __int64 v8; // r14
  __int64 v9; // r12
  unsigned int v11; // r14d
  unsigned int *v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rsi
  unsigned int *v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rsi
  unsigned int v20; // r14d
  __int64 v21; // [rsp+0h] [rbp-70h]
  __int64 v22; // [rsp+8h] [rbp-68h]
  _BYTE v23[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v24; // [rsp+30h] [rbp-40h]

  if ( (unsigned int)(a2 - 13) <= 0x11 )
  {
    v21 = a3[1];
    v22 = *a3;
    v9 = (*(__int64 (__fastcall **)(unsigned int *, __int64, _QWORD, __int64))(*(_QWORD *)a1[10] + 16LL))(
           a1[10],
           a2,
           *a3,
           v21);
    if ( !v9 )
    {
      v24 = 257;
      v9 = sub_B504D0((unsigned int)a2, v22, v21, v23, 0, 0);
      if ( (unsigned __int8)sub_920620(v9) )
      {
        v11 = *((_DWORD *)a1 + 26);
        if ( a6 || (a6 = a1[12]) != 0 )
          sub_B99FD0(v9, 3, a6);
        sub_B45150(v9, v11);
      }
      (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
        a1[11],
        v9,
        a5,
        a1[7],
        a1[8]);
      v12 = *a1;
      v13 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
      if ( *a1 != (unsigned int *)v13 )
      {
        do
        {
          v14 = *((_QWORD *)v12 + 1);
          v15 = *v12;
          v12 += 4;
          sub_B99FD0(v9, v15, v14);
        }
        while ( (unsigned int *)v13 != v12 );
      }
    }
  }
  else
  {
    if ( (_DWORD)a2 != 12 )
      BUG();
    v8 = *a3;
    v9 = (*(__int64 (__fastcall **)(unsigned int *, __int64, _QWORD, _QWORD))(*(_QWORD *)a1[10] + 48LL))(
           a1[10],
           12,
           *a3,
           *((unsigned int *)a1 + 26));
    if ( !v9 )
    {
      v24 = 257;
      v9 = sub_B50340(12, v8, v23, 0, 0);
      if ( (unsigned __int8)sub_920620(v9) )
      {
        v20 = *((_DWORD *)a1 + 26);
        if ( a6 || (a6 = a1[12]) != 0 )
          sub_B99FD0(v9, 3, a6);
        sub_B45150(v9, v20);
      }
      (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
        a1[11],
        v9,
        a5,
        a1[7],
        a1[8]);
      v16 = *a1;
      v17 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
      if ( *a1 != (unsigned int *)v17 )
      {
        do
        {
          v18 = *((_QWORD *)v16 + 1);
          v19 = *v16;
          v16 += 4;
          sub_B99FD0(v9, v19, v18);
        }
        while ( (unsigned int *)v17 != v16 );
      }
    }
  }
  return v9;
}
