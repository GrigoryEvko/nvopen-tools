// Function: sub_2416060
// Address: 0x2416060
//
void __fastcall sub_2416060(__int64 *a1, __int64 a2, _BYTE *a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 *v7; // rax
  _QWORD v8[2]; // [rsp+0h] [rbp-F0h] BYREF
  _BYTE v9[32]; // [rsp+10h] [rbp-E0h] BYREF
  __int16 v10; // [rsp+30h] [rbp-C0h]
  unsigned int *v11[2]; // [rsp+40h] [rbp-B0h] BYREF
  char v12; // [rsp+50h] [rbp-A0h] BYREF
  void *v13; // [rsp+C0h] [rbp-30h]

  sub_23D0AB0((__int64)v11, a2, 0, 0, 0);
  v8[0] = sub_24159D0((__int64)a1, (__int64)a3);
  if ( (unsigned __int8)sub_240D530() )
  {
    v8[1] = sub_2414930(a1, a3);
    v4 = *a1;
    v10 = 257;
    v5 = sub_921880(v11, *(_QWORD *)(v4 + 456), *(_QWORD *)(v4 + 464), (int)v8, 2, (__int64)v9, 0);
  }
  else
  {
    v10 = 257;
    v5 = sub_921880(v11, *(_QWORD *)(*a1 + 440), *(_QWORD *)(*a1 + 448), (int)v8, 1, (__int64)v9, 0);
  }
  v6 = v5;
  v7 = (__int64 *)sub_BD5C60(v5);
  *(_QWORD *)(v6 + 72) = sub_A7A090((__int64 *)(v6 + 72), v7, 1, 79);
  nullsub_61();
  v13 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v11[0] != &v12 )
    _libc_free((unsigned __int64)v11[0]);
}
