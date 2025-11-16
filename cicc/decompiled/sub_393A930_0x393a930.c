// Function: sub_393A930
// Address: 0x393a930
//
__int64 *__fastcall sub_393A930(__int64 *a1, _QWORD *a2, _DWORD **a3)
{
  _QWORD *v3; // r15
  unsigned __int64 *v6; // rax
  unsigned __int64 *v7; // rcx
  __int64 v8; // rdx
  unsigned __int64 *v9; // rcx
  __int64 v10; // rax
  unsigned __int64 *v11; // [rsp+0h] [rbp-80h]
  unsigned __int64 *v12; // [rsp+18h] [rbp-68h]
  unsigned __int64 v13; // [rsp+28h] [rbp-58h]
  unsigned __int64 v14; // [rsp+30h] [rbp-50h] BYREF
  __int64 v15; // [rsp+38h] [rbp-48h] BYREF
  __int64 v16; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 v17[7]; // [rsp+48h] [rbp-38h] BYREF

  v3 = (_QWORD *)(*a2 & 0xFFFFFFFFFFFFFFFELL);
  if ( v3 )
  {
    *a2 = 0;
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD *, void *))(*v3 + 48LL))(v3, &unk_4FA032A) )
    {
      v6 = (unsigned __int64 *)v3[2];
      v7 = (unsigned __int64 *)v3[1];
      v13 = 1;
      v11 = v6;
      if ( v6 == v7 )
      {
        v10 = 1;
      }
      else
      {
        do
        {
          v12 = v7;
          v14 = *v7;
          *v7 = 0;
          sub_393A8B0(&v15, &v14, a3);
          v16 = v13 | 1;
          sub_12BEC00(v17, (unsigned __int64 *)&v16, (unsigned __int64 *)&v15);
          v9 = v12;
          v13 = v17[0] | 1;
          if ( (v16 & 1) != 0 || (v16 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_16BCAE0(&v16, (__int64)&v16, v8);
          if ( (v15 & 1) != 0 || (v15 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_16BCAE0(&v15, (__int64)&v16, v8);
          if ( v14 )
          {
            (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v14 + 8LL))(v14);
            v9 = v12;
          }
          v7 = v9 + 1;
        }
        while ( v11 != v7 );
        v10 = v13 | 1;
      }
      *a1 = v10;
      (*(void (__fastcall **)(_QWORD *))(*v3 + 8LL))(v3);
    }
    else
    {
      v17[0] = (unsigned __int64)v3;
      sub_393A8B0(a1, v17, a3);
      if ( v17[0] )
        (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v17[0] + 8LL))(v17[0]);
    }
  }
  else
  {
    *a2 = 0;
    *a1 = 1;
  }
  return a1;
}
