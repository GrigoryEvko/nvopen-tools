// Function: sub_1D29D50
// Address: 0x1d29d50
//
_QWORD *__fastcall sub_1D29D50(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  int v8; // ebx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned int v11; // ebx
  int v12; // eax
  unsigned int v13; // eax
  __int64 v14; // r15
  __int64 v15; // r13
  __int64 v16; // r14
  unsigned int v17; // r13d
  unsigned int v18; // eax
  int v19; // r13d
  __int64 v20; // rax
  unsigned int v21; // edx
  unsigned __int8 v22; // al
  __int64 v24; // [rsp+0h] [rbp-50h] BYREF
  __int64 v25; // [rsp+8h] [rbp-48h]
  _QWORD v26[8]; // [rsp+10h] [rbp-40h] BYREF

  v26[0] = a2;
  v26[1] = a3;
  v24 = a4;
  v25 = a5;
  if ( (_BYTE)a4 )
    v8 = sub_1D13440(a4);
  else
    v8 = sub_1F58D40(&v24, a2, a3, a4, a5, a6);
  v11 = (unsigned int)(v8 + 7) >> 3;
  if ( LOBYTE(v26[0]) )
    v12 = sub_1D13440(v26[0]);
  else
    v12 = sub_1F58D40(v26, a2, v6, v7, v9, v10);
  v13 = (unsigned int)(v12 + 7) >> 3;
  if ( v13 >= v11 )
    v11 = v13;
  v14 = sub_1F58E60(v26, a1[6]);
  v15 = sub_1F58E60(&v24, a1[6]);
  v16 = sub_1E0A0C0(a1[4]);
  v17 = sub_15AAE50(v16, v15);
  v18 = sub_15AAE50(v16, v14);
  if ( v17 >= v18 )
    v18 = v17;
  v19 = sub_1E090F0(*(_QWORD *)(a1[4] + 56LL), v11, v18, 0, 0, 0, v24, v25);
  v20 = sub_1E0A0C0(a1[4]);
  v21 = 8 * sub_15A9520(v20, *(_DWORD *)(v20 + 4));
  if ( v21 == 32 )
  {
    v22 = 5;
  }
  else if ( v21 > 0x20 )
  {
    v22 = 6;
    if ( v21 != 64 )
    {
      v22 = 0;
      if ( v21 == 128 )
        v22 = 7;
    }
  }
  else
  {
    v22 = 3;
    if ( v21 != 8 )
      v22 = 4 * (v21 == 16);
  }
  return sub_1D299D0(a1, v19, v22, 0, 0);
}
