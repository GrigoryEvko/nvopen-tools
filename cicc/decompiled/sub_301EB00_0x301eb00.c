// Function: sub_301EB00
// Address: 0x301eb00
//
_QWORD *__fastcall sub_301EB00(_QWORD *a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  _QWORD *v8; // r12
  __int128 v10; // [rsp-70h] [rbp-B0h]
  __int128 v11; // [rsp-60h] [rbp-A0h]
  __int128 v12; // [rsp-50h] [rbp-90h]
  __int128 v13; // [rsp-40h] [rbp-80h]

  v7 = sub_22077B0(0x130u);
  v8 = (_QWORD *)v7;
  if ( v7 )
  {
    *((_QWORD *)&v13 + 1) = 45;
    *(_QWORD *)&v13 = &off_49D4660;
    *((_QWORD *)&v12 + 1) = 86;
    *(_QWORD *)&v12 = &off_49D5740;
    *((_QWORD *)&v11 + 1) = 45;
    *(_QWORD *)&v11 = qword_502A920;
    *((_QWORD *)&v10 + 1) = a5;
    *(_QWORD *)&v10 = a4;
    sub_EA0E70(
      v7,
      a1,
      a2,
      a3,
      a2,
      a3,
      v10,
      v11,
      v12,
      v13,
      (__int64)&unk_4458D80,
      (__int64)&unk_4458D7C,
      (__int64)&unk_4458D70,
      0,
      0,
      0);
    *v8 = &unk_4A2E228;
  }
  return v8;
}
