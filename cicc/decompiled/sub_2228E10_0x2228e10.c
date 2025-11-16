// Function: sub_2228E10
// Address: 0x2228e10
//
_QWORD *__fastcall sub_2228E10(
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
  __int64 v14; // rdx
  __int64 v15; // rax
  _QWORD *v16; // r14
  unsigned int v17; // edx
  unsigned int v18; // eax
  _QWORD *v19; // r8
  unsigned __int64 v20; // rdx
  __int64 v21; // rcx
  unsigned __int64 v22; // rsi
  char v23; // bl
  char v24; // r15
  __int64 v25; // rcx
  int *v27; // rax
  int v28; // eax
  unsigned int *v29; // rax
  _QWORD *v30; // [rsp+0h] [rbp-128h]
  unsigned __int64 v31; // [rsp+0h] [rbp-128h]
  unsigned int v32; // [rsp+28h] [rbp-100h] BYREF
  unsigned int v33; // [rsp+2Ch] [rbp-FCh] BYREF
  _QWORD v34[31]; // [rsp+30h] [rbp-F8h] BYREF

  v12 = *(_QWORD **)(sub_2244AF0(a6 + 208) + 16);
  v34[0] = v12[37];
  v34[1] = v12[38];
  v34[2] = v12[39];
  v34[3] = v12[40];
  v34[4] = v12[41];
  v34[5] = v12[42];
  v34[6] = v12[43];
  v34[7] = v12[44];
  v34[8] = v12[45];
  v34[9] = v12[46];
  v34[10] = v12[47];
  v34[11] = v12[48];
  v34[12] = v12[25];
  v34[13] = v12[26];
  v34[14] = v12[27];
  v34[15] = v12[28];
  v13 = v12[29];
  v33 = 0;
  v34[16] = v13;
  v34[17] = v12[30];
  v34[18] = v12[31];
  v34[19] = v12[32];
  v34[20] = v12[33];
  v34[21] = v12[34];
  v14 = v12[35];
  v15 = v12[36];
  v34[22] = v14;
  v34[23] = v15;
  v16 = sub_2228670(a1, a2, a3, a4, a5, &v32, (__int64)v34, 12, a6, &v33);
  v18 = v17;
  v19 = v16;
  v20 = v17 | a3 & 0xFFFFFFFF00000000LL;
  v21 = v33;
  if ( v33 )
  {
    v22 = (unsigned __int64)a7;
    *a7 |= 4u;
  }
  else
  {
    v21 = a8;
    v22 = v32;
    *(_DWORD *)(a8 + 16) = v32;
  }
  v23 = v18 == -1;
  v24 = v23 & (v16 != 0);
  if ( v24 )
  {
    v29 = (unsigned int *)v16[2];
    if ( (unsigned __int64)v29 >= v16[3] )
    {
      v31 = v20;
      v18 = (*(__int64 (__fastcall **)(_QWORD *, unsigned __int64, unsigned __int64, __int64, _QWORD *))(*v16 + 72LL))(
              v16,
              v22,
              v20,
              v21,
              v16);
      v20 = v31;
    }
    else
    {
      v18 = *v29;
    }
    v23 = 0;
    v19 = 0;
    if ( v18 == -1 )
      v23 = v24;
    else
      v19 = v16;
  }
  LOBYTE(v18) = a5 == -1;
  v25 = v18;
  if ( a4 && a5 == -1 )
  {
    v27 = (int *)a4[2];
    if ( (unsigned __int64)v27 >= a4[3] )
    {
      v30 = v19;
      v28 = (*(__int64 (__fastcall **)(_QWORD *, unsigned __int64, unsigned __int64, __int64))(*a4 + 72LL))(
              a4,
              v22,
              v20,
              v25);
      v19 = v30;
    }
    else
    {
      v28 = *v27;
    }
    LOBYTE(v25) = v28 == -1;
  }
  if ( v23 == (_BYTE)v25 )
    *a7 |= 2u;
  return v19;
}
