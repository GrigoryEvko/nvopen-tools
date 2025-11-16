// Function: sub_161A730
// Address: 0x161a730
//
__int64 __fastcall sub_161A730(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // r8
  unsigned __int64 v3; // r8
  _QWORD *v5; // r15
  unsigned __int64 *v6; // rax
  unsigned __int64 *v7; // rcx
  __int64 v8; // rax
  unsigned __int64 *v9; // rcx
  __int64 v10; // rax
  unsigned __int64 *v11; // [rsp+8h] [rbp-98h]
  unsigned __int64 *v12; // [rsp+20h] [rbp-80h]
  __int64 v13; // [rsp+38h] [rbp-68h] BYREF
  __int64 v14; // [rsp+40h] [rbp-60h] BYREF
  __int64 v15; // [rsp+48h] [rbp-58h] BYREF
  unsigned __int64 v16; // [rsp+50h] [rbp-50h] BYREF
  __int64 v17; // [rsp+58h] [rbp-48h] BYREF
  unsigned __int64 v18; // [rsp+60h] [rbp-40h] BYREF
  unsigned __int64 v19[7]; // [rsp+68h] [rbp-38h] BYREF

  sub_15E4B20((__int64)&v13, a2);
  v2 = v13;
  v13 = 0;
  v3 = v2 & 0xFFFFFFFFFFFFFFFELL;
  if ( v3 )
  {
    v14 = 0;
    v5 = (_QWORD *)v3;
    if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v3 + 48LL))(v3, &unk_4FA032A) )
    {
      v6 = (unsigned __int64 *)v5[2];
      v7 = (unsigned __int64 *)v5[1];
      v15 = 1;
      v11 = v6;
      if ( v7 == v6 )
      {
        v10 = 1;
      }
      else
      {
        do
        {
          v12 = v7;
          v18 = *v7;
          *v7 = 0;
          sub_160DD30(&v17, &v18);
          v8 = v15;
          v15 = 0;
          v19[0] = v8 | 1;
          sub_12BEC00(&v16, v19, (unsigned __int64 *)&v17);
          if ( (v15 & 1) != 0 || (v9 = v12, (v15 & 0xFFFFFFFFFFFFFFFELL) != 0) )
            sub_16BCAE0(&v15);
          v15 |= v16 | 1;
          if ( (v19[0] & 1) != 0 || (v19[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_16BCAE0(v19);
          if ( (v17 & 1) != 0 || (v17 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_16BCAE0(&v17);
          if ( v18 )
          {
            (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v18 + 8LL))(v18);
            v9 = v12;
          }
          v7 = v9 + 1;
        }
        while ( v11 != v7 );
        v10 = v15 | 1;
      }
      v18 = v10;
      (*(void (__fastcall **)(_QWORD *))(*v5 + 8LL))(v5);
    }
    else
    {
      v19[0] = (unsigned __int64)v5;
      sub_160DD30((__int64 *)&v18, v19);
      if ( v19[0] )
        (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v19[0] + 8LL))(v19[0]);
    }
    if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v18 = v18 & 0xFFFFFFFFFFFFFFFELL | 1;
      sub_16BCAE0(&v18);
    }
    if ( (v14 & 1) != 0 || (v14 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_16BCAE0(&v14);
  }
  if ( (v13 & 1) != 0 || (v13 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(&v13);
  return sub_161A600(*(_QWORD *)(a1 + 8), a2);
}
