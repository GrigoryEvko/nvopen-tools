// Function: sub_161ACF0
// Address: 0x161acf0
//
void __fastcall sub_161ACF0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // r13
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // [rsp+0h] [rbp-60h]
  __int64 v12; // [rsp+8h] [rbp-58h] BYREF
  __int64 *v13; // [rsp+10h] [rbp-50h]
  __int64 v14; // [rsp+18h] [rbp-48h]
  __int64 v15[8]; // [rsp+20h] [rbp-40h] BYREF

  v12 = a2;
  if ( a3 )
  {
    v4 = *(_QWORD *)sub_161A990(a1 + 568, (unsigned __int64 *)&v12);
    if ( v4 )
    {
      v5 = v4 + 568;
    }
    else
    {
      v8 = sub_22077B0(1312);
      v9 = v8;
      if ( v8 )
      {
        v11 = v8;
        sub_16113E0(v8);
        v9 = v11;
        v10 = v11 + 568;
        v5 = v11 + 568;
      }
      else
      {
        v5 = 568;
        v10 = 0;
      }
      *(_QWORD *)(v9 + 176) = v10;
      *(_QWORD *)sub_161A990(a1 + 568, (unsigned __int64 *)&v12) = v9;
    }
    v6 = sub_1614F20(*(_QWORD *)(a1 + 176), a3[2]);
    if ( !v6 || !*(_BYTE *)(v6 + 41) || (v7 = sub_160EA80(v5, a3[2])) == 0 )
    {
      sub_16185C0(v5, a3);
      v7 = (__int64)a3;
    }
    v15[0] = v7;
    v13 = v15;
    v14 = 0x100000001LL;
    sub_1613D20(v5, v15, 1, v12);
    if ( v13 != v15 )
      _libc_free((unsigned __int64)v13);
  }
}
