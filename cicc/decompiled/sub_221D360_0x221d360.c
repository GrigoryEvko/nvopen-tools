// Function: sub_221D360
// Address: 0x221d360
//
_QWORD *__fastcall sub_221D360(
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
  char v24; // al
  int v26; // eax
  int v27; // eax
  _QWORD *v28; // [rsp+0h] [rbp-D8h]
  unsigned int v29; // [rsp+28h] [rbp-B0h] BYREF
  unsigned int v30; // [rsp+2Ch] [rbp-ACh] BYREF
  _QWORD v31[21]; // [rsp+30h] [rbp-A8h] BYREF

  v12 = *(_QWORD **)(sub_22311C0(a6 + 208) + 16);
  v30 = 0;
  v31[0] = v12[18];
  v31[1] = v12[19];
  v31[2] = v12[20];
  v31[3] = v12[21];
  v31[4] = v12[22];
  v31[5] = v12[23];
  v31[6] = v12[24];
  v31[7] = v12[11];
  v31[8] = v12[12];
  v31[9] = v12[13];
  v31[10] = v12[14];
  v31[11] = v12[15];
  v13 = v12[16];
  v14 = v12[17];
  v31[12] = v13;
  v31[13] = v14;
  v15 = sub_221CDA0(a1, a2, a3, a4, a5, &v29, (__int64)v31, 7, a6, &v30);
  v17 = v16;
  v18 = v15;
  v19 = v16 | a3 & 0xFFFFFFFF00000000LL;
  v20 = v30;
  if ( v30 )
  {
    v21 = (unsigned __int64)a7;
    *a7 |= 4u;
  }
  else
  {
    v20 = a8;
    v21 = v29;
    *(_DWORD *)(a8 + 24) = v29;
  }
  v22 = v17 == -1;
  v23 = v22 & (v15 != 0);
  if ( v23 )
  {
    v22 = 0;
    if ( v15[2] >= v15[3] )
    {
      v27 = (*(__int64 (__fastcall **)(_QWORD *, unsigned __int64, unsigned __int64, __int64, _QWORD *))(*v15 + 72LL))(
              v15,
              v21,
              v19,
              v20,
              v15);
      v18 = 0;
      if ( v27 == -1 )
        v22 = v23;
      else
        v18 = v15;
    }
  }
  v24 = a5 == -1;
  if ( a4 )
  {
    if ( a5 == -1 )
    {
      v24 = 0;
      if ( a4[2] >= a4[3] )
      {
        v28 = v18;
        v26 = (*(__int64 (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4);
        v18 = v28;
        v24 = v26 == -1;
      }
    }
  }
  if ( v22 == v24 )
    *a7 |= 2u;
  return v18;
}
