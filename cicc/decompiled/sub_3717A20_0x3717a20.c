// Function: sub_3717A20
// Address: 0x3717a20
//
__int64 __fastcall sub_3717A20(
        __int64 a1,
        unsigned __int16 a2,
        unsigned __int64 a3,
        char *a4,
        signed __int64 a5,
        __int64 a6,
        unsigned int a7,
        __int64 a8,
        unsigned __int64 a9,
        const char *a10,
        __int64 a11)
{
  const char *v13; // rax
  int v14; // eax
  _QWORD *v15; // r12
  __int64 v17; // [rsp+10h] [rbp-98h]
  _QWORD *v18; // [rsp+18h] [rbp-90h]
  const char *v19; // [rsp+20h] [rbp-88h]
  __int64 v20[2]; // [rsp+28h] [rbp-80h] BYREF
  __int64 v21; // [rsp+38h] [rbp-70h] BYREF
  const char *v22; // [rsp+48h] [rbp-60h] BYREF
  __int64 v23; // [rsp+50h] [rbp-58h]
  const char *v24; // [rsp+58h] [rbp-50h]
  signed __int64 v25; // [rsp+60h] [rbp-48h]
  __int16 v26; // [rsp+68h] [rbp-40h]

  v17 = sub_3717770((__int64 **)a1, a2, a3, a4, a5, a6, a7, a8, a9);
  v13 = ".offloading.entry.";
  if ( (unsigned int)(*(_DWORD *)(a1 + 264) - 42) < 2 )
    v13 = "$offloading$entry$";
  v19 = v13;
  v18 = (_QWORD *)sub_3717640((__int64 **)a1);
  v14 = *(_DWORD *)(a1 + 324);
  v25 = a5;
  LODWORD(v20[0]) = v14;
  v22 = v19;
  v23 = 18;
  v24 = a4;
  v26 = 1285;
  BYTE4(v20[0]) = 1;
  v15 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v15 )
    sub_B30000((__int64)v15, a1, v18, 1, 4, v17, (__int64)&v22, 0, 0, v20[0], 0);
  if ( *(_DWORD *)(a1 + 284) == 1 )
  {
    v22 = a10;
    v26 = 773;
    v24 = "$OE";
    v23 = a11;
    sub_CA0F50(v20, (void **)&v22);
    sub_B31A00((__int64)v15, v20[0], v20[1]);
    if ( (__int64 *)v20[0] != &v21 )
      j_j___libc_free_0(v20[0]);
  }
  else
  {
    sub_B31A00((__int64)v15, (__int64)a10, a11);
  }
  return sub_B2F770((__int64)v15, 3u);
}
