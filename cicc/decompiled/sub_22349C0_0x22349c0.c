// Function: sub_22349C0
// Address: 0x22349c0
//
__int64 *__fastcall sub_22349C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        _DWORD *a7,
        __int64 a8)
{
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // r14
  unsigned int v17; // edx
  unsigned int v18; // eax
  __int64 *v19; // r8
  unsigned __int64 v20; // rdx
  __int64 v21; // rcx
  unsigned __int64 v22; // rsi
  char v23; // bl
  char v24; // r15
  char v25; // al
  int v27; // eax
  int v28; // eax
  __int64 *v29; // [rsp+0h] [rbp-128h]
  unsigned int v30; // [rsp+28h] [rbp-100h] BYREF
  unsigned int v31; // [rsp+2Ch] [rbp-FCh] BYREF
  _QWORD v32[31]; // [rsp+30h] [rbp-F8h] BYREF

  v12 = *(_QWORD **)(sub_22311C0((_QWORD *)(a6 + 208), a2) + 16);
  v32[0] = v12[37];
  v32[1] = v12[38];
  v32[2] = v12[39];
  v32[3] = v12[40];
  v32[4] = v12[41];
  v32[5] = v12[42];
  v32[6] = v12[43];
  v32[7] = v12[44];
  v32[8] = v12[45];
  v32[9] = v12[46];
  v32[10] = v12[47];
  v32[11] = v12[48];
  v32[12] = v12[25];
  v32[13] = v12[26];
  v32[14] = v12[27];
  v32[15] = v12[28];
  v13 = v12[29];
  v31 = 0;
  v32[16] = v13;
  v32[17] = v12[30];
  v32[18] = v12[31];
  v32[19] = v12[32];
  v32[20] = v12[33];
  v32[21] = v12[34];
  v14 = v12[35];
  v15 = v12[36];
  v32[22] = v14;
  v32[23] = v15;
  v16 = sub_22343F0(a1, a2, a3, a4, a5, &v30, (__int64)v32, 12, a6, &v31);
  v18 = v17;
  v19 = v16;
  v20 = v17 | a3 & 0xFFFFFFFF00000000LL;
  v21 = v31;
  if ( v31 )
  {
    v22 = (unsigned __int64)a7;
    *a7 |= 4u;
  }
  else
  {
    v21 = a8;
    v22 = v30;
    *(_DWORD *)(a8 + 16) = v30;
  }
  v23 = v18 == -1;
  v24 = v23 & (v16 != 0);
  if ( v24 )
  {
    v23 = 0;
    if ( v16[2] >= (unsigned __int64)v16[3] )
    {
      v28 = (*(__int64 (__fastcall **)(__int64 *, unsigned __int64, unsigned __int64, __int64, __int64 *))(*v16 + 72))(
              v16,
              v22,
              v20,
              v21,
              v16);
      v19 = 0;
      if ( v28 == -1 )
        v23 = v24;
      else
        v19 = v16;
    }
  }
  v25 = (_DWORD)a5 == -1;
  if ( a4 )
  {
    if ( (_DWORD)a5 == -1 )
    {
      v25 = 0;
      if ( a4[2] >= (unsigned __int64)a4[3] )
      {
        v29 = v19;
        v27 = (*(__int64 (__fastcall **)(__int64 *))(*a4 + 72))(a4);
        v19 = v29;
        v25 = v27 == -1;
      }
    }
  }
  if ( v23 == v25 )
    *a7 |= 2u;
  return v19;
}
