// Function: sub_2D4B980
// Address: 0x2d4b980
//
_BOOL8 __fastcall sub_2D4B980(
        _QWORD **a1,
        unsigned int **a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        int a7,
        char a8,
        __int64 *a9,
        __int64 *a10,
        __int64 a11)
{
  __int16 v14; // r11
  _QWORD *v15; // r15
  __int64 v16; // rbx
  unsigned int *v17; // r14
  __int64 v18; // rdx
  unsigned int v19; // esi
  __int64 v20; // rax
  __int16 v23; // [rsp+10h] [rbp-A0h]
  int v24; // [rsp+14h] [rbp-9Ch]
  int v25; // [rsp+4Ch] [rbp-64h] BYREF
  _QWORD v26[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v27; // [rsp+70h] [rbp-40h]

  switch ( a7 )
  {
    case 2:
    case 5:
      v14 = 2;
      break;
    case 4:
    case 6:
      v14 = 4;
      break;
    case 7:
      v14 = a7;
      break;
    default:
      BUG();
  }
  v27 = 257;
  v23 = v14;
  v24 = a6;
  v15 = sub_BD2C40(80, unk_3F148C4);
  if ( v15 )
    sub_B4D5A0((__int64)v15, a3, a4, a5, v24, a7, v23, a8, 0, 0);
  (*(void (__fastcall **)(unsigned int *, _QWORD *, _QWORD *, unsigned int *, unsigned int *))(*(_QWORD *)a2[11] + 16LL))(
    a2[11],
    v15,
    v26,
    a2[7],
    a2[8]);
  v16 = (__int64)&(*a2)[4 * *((unsigned int *)a2 + 2)];
  if ( *a2 != (unsigned int *)v16 )
  {
    v17 = *a2;
    do
    {
      v18 = *((_QWORD *)v17 + 1);
      v19 = *v17;
      v17 += 4;
      sub_B99FD0((__int64)v15, v19, v18);
    }
    while ( (unsigned int *)v16 != v17 );
  }
  if ( a11 )
    sub_2D42CA0((__int64)v15, a11);
  v26[0] = "success";
  v27 = 259;
  v25 = 1;
  v20 = sub_94D3D0(a2, (__int64)v15, (__int64)&v25, 1, (__int64)v26);
  v27 = 259;
  *a9 = v20;
  v26[0] = "newloaded";
  v25 = 0;
  *a10 = sub_94D3D0(a2, (__int64)v15, (__int64)&v25, 1, (__int64)v26);
  return sub_2D4B8E0(*a1, (__int64)v15);
}
