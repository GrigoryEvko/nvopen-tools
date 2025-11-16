// Function: sub_2228BE0
// Address: 0x2228be0
//
_QWORD *__fastcall sub_2228BE0(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        _QWORD *a4,
        int a5,
        __int64 a6,
        _DWORD *a7,
        __int64 a8)
{
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  _QWORD *v15; // r14
  unsigned int v16; // edx
  unsigned int v17; // eax
  _QWORD *v18; // r8
  unsigned __int64 v19; // rdx
  __int64 v20; // rcx
  unsigned __int64 v21; // rsi
  char v22; // bl
  char v23; // r15
  __int64 v24; // rcx
  int *v26; // rax
  int v27; // eax
  unsigned int *v28; // rax
  _QWORD *v29; // [rsp+0h] [rbp-D8h]
  unsigned __int64 v30; // [rsp+0h] [rbp-D8h]
  unsigned int v31; // [rsp+28h] [rbp-B0h] BYREF
  unsigned int v32; // [rsp+2Ch] [rbp-ACh] BYREF
  _QWORD v33[21]; // [rsp+30h] [rbp-A8h] BYREF

  v12 = *(_QWORD **)(sub_2244AF0(a6 + 208) + 16);
  v32 = 0;
  v33[0] = v12[18];
  v33[1] = v12[19];
  v33[2] = v12[20];
  v33[3] = v12[21];
  v33[4] = v12[22];
  v33[5] = v12[23];
  v33[6] = v12[24];
  v33[7] = v12[11];
  v33[8] = v12[12];
  v33[9] = v12[13];
  v33[10] = v12[14];
  v33[11] = v12[15];
  v13 = v12[16];
  v14 = v12[17];
  v33[12] = v13;
  v33[13] = v14;
  v15 = sub_2228670(a1, a2, a3, a4, a5, &v31, (__int64)v33, 7, a6, &v32);
  v17 = v16;
  v18 = v15;
  v19 = v16 | a3 & 0xFFFFFFFF00000000LL;
  v20 = v32;
  if ( v32 )
  {
    v21 = (unsigned __int64)a7;
    *a7 |= 4u;
  }
  else
  {
    v20 = a8;
    v21 = v31;
    *(_DWORD *)(a8 + 24) = v31;
  }
  v22 = v17 == -1;
  v23 = v22 & (v15 != 0);
  if ( v23 )
  {
    v28 = (unsigned int *)v15[2];
    if ( (unsigned __int64)v28 >= v15[3] )
    {
      v30 = v19;
      v17 = (*(__int64 (__fastcall **)(_QWORD *, unsigned __int64, unsigned __int64, __int64, _QWORD *))(*v15 + 72LL))(
              v15,
              v21,
              v19,
              v20,
              v15);
      v19 = v30;
    }
    else
    {
      v17 = *v28;
    }
    v22 = 0;
    v18 = 0;
    if ( v17 == -1 )
      v22 = v23;
    else
      v18 = v15;
  }
  LOBYTE(v17) = a5 == -1;
  v24 = v17;
  if ( a4 && a5 == -1 )
  {
    v26 = (int *)a4[2];
    if ( (unsigned __int64)v26 >= a4[3] )
    {
      v29 = v18;
      v27 = (*(__int64 (__fastcall **)(_QWORD *, unsigned __int64, unsigned __int64, __int64))(*a4 + 72LL))(
              a4,
              v21,
              v19,
              v24);
      v18 = v29;
    }
    else
    {
      v27 = *v26;
    }
    LOBYTE(v24) = v27 == -1;
  }
  if ( v22 == (_BYTE)v24 )
    *a7 |= 2u;
  return v18;
}
