// Function: sub_3532D60
// Address: 0x3532d60
//
_QWORD *__fastcall sub_3532D60(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // rsi
  __int64 v5; // rdx
  __int64 (__fastcall ***v6)(_QWORD); // r15
  __int64 v7; // r14
  _QWORD *v8; // r12
  int v9; // eax
  unsigned int v10; // ebx
  __int64 v11; // r15
  int v12; // eax
  int v13; // ebx
  int v14; // eax
  __int64 v16; // [rsp+8h] [rbp-48h]
  _QWORD *v17; // [rsp+10h] [rbp-40h]
  unsigned int v18; // [rsp+1Ch] [rbp-34h]

  v3 = a2 - (_QWORD)a1;
  v5 = v3 >> 3;
  v17 = a1;
  if ( v3 > 0 )
  {
    do
    {
      v6 = (__int64 (__fastcall ***)(_QWORD))*a3;
      v16 = v5;
      v7 = v5 >> 1;
      v8 = &v17[v5 >> 1];
      v9 = (**(__int64 (__fastcall ***)(_QWORD))*a3)(*a3);
      LODWORD(v6) = *((_DWORD *)v6 + 10);
      v10 = (_DWORD)v6 * (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*v8 + 8LL))(*v8) * v9;
      v11 = *v8;
      v18 = v10;
      v12 = (**(__int64 (__fastcall ***)(_QWORD))*v8)(*v8);
      LODWORD(v11) = *(_DWORD *)(v11 + 40);
      v13 = v12;
      v14 = (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*a3 + 8LL))(*a3);
      v5 = v7;
      if ( v18 <= v14 * v13 * (int)v11 )
      {
        v17 = v8 + 1;
        v5 = v16 - v7 - 1;
      }
    }
    while ( v5 > 0 );
  }
  return v17;
}
