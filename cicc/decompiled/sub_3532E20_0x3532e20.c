// Function: sub_3532E20
// Address: 0x3532e20
//
_QWORD *__fastcall sub_3532E20(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // rsi
  __int64 v5; // rdx
  _QWORD *v6; // r14
  __int64 (__fastcall ***v7)(_QWORD); // r15
  int v8; // eax
  unsigned int v9; // r12d
  __int64 v10; // r15
  int v11; // eax
  __int64 v13; // [rsp+8h] [rbp-48h]
  _QWORD *v14; // [rsp+10h] [rbp-40h]
  __int64 v15; // [rsp+18h] [rbp-38h]

  v3 = a2 - (_QWORD)a1;
  v5 = v3 >> 3;
  v14 = a1;
  if ( v3 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v13 = v5;
        v6 = &v14[v5 >> 1];
        v15 = v5 >> 1;
        v7 = (__int64 (__fastcall ***)(_QWORD))*v6;
        v8 = (**(__int64 (__fastcall ***)(_QWORD))*v6)(*v6);
        LODWORD(v7) = *((_DWORD *)v7 + 10);
        v9 = (_DWORD)v7 * (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*a3 + 8LL))(*a3) * v8;
        v10 = *a3;
        v11 = (**(__int64 (__fastcall ***)(_QWORD))*a3)(*a3);
        LODWORD(v10) = *(_DWORD *)(v10 + 40);
        if ( v9 <= (unsigned int)v10 * (*(unsigned int (__fastcall **)(_QWORD))(*(_QWORD *)*v6 + 8LL))(*v6) * v11 )
          break;
        v14 = v6 + 1;
        v5 = v13 - v15 - 1;
        if ( v5 <= 0 )
          return v14;
      }
      v5 = v15;
    }
    while ( v15 > 0 );
  }
  return v14;
}
