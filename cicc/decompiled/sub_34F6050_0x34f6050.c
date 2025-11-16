// Function: sub_34F6050
// Address: 0x34f6050
//
_QWORD *__fastcall sub_34F6050(__int64 *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  _QWORD *v8; // rax
  _QWORD *v9; // r12
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 (*v12)(void); // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 (*v20)(void); // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9

  v8 = (_QWORD *)sub_22077B0(0x328u);
  v9 = v8;
  if ( v8 )
  {
    v10 = a2[2];
    v8[1] = a2;
    *v8 = off_4A38700;
    v11 = *a1;
    v9[4] = a3;
    v9[2] = v11;
    v9[3] = a1[1];
    v9[5] = a2[4];
    v12 = *(__int64 (**)(void))(*(_QWORD *)v10 + 128LL);
    v13 = 0;
    if ( v12 != sub_2DAC790 )
    {
      v13 = v12();
      v10 = a2[2];
    }
    v9[6] = v13;
    v14 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 200LL))(v10);
    v15 = a2[2];
    v9[8] = 0;
    v9[7] = v14;
    v9[11] = v9 + 13;
    v9[12] = 0x800000000LL;
    v9[18] = 0x800000000LL;
    v9[48] = 0x800000000LL;
    v9[57] = off_4A38748;
    v16 = *a1;
    v9[17] = v9 + 19;
    v9[59] = v16;
    v17 = a1[1];
    v9[24] = v9 + 27;
    v9[60] = v17;
    v18 = a1[2];
    v9[36] = v9 + 39;
    v9[61] = v18;
    v19 = a2[4];
    v9[47] = v9 + 49;
    v9[9] = 0;
    *((_DWORD *)v9 + 21) = 0;
    v9[23] = 0;
    v9[25] = 8;
    *((_DWORD *)v9 + 52) = 0;
    *((_BYTE *)v9 + 212) = 1;
    v9[35] = 0;
    v9[37] = 8;
    *((_DWORD *)v9 + 76) = 0;
    *((_BYTE *)v9 + 308) = 1;
    v9[58] = a2;
    v9[62] = a3;
    v9[63] = v19;
    v20 = *(__int64 (**)(void))(*(_QWORD *)v15 + 128LL);
    v21 = 0;
    if ( v20 != sub_2DAC790 )
    {
      v21 = v20();
      v15 = a2[2];
    }
    v9[64] = v21;
    v22 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v15 + 200LL))(v15);
    v23 = a2[13];
    v24 = v9[59];
    v9[65] = v22;
    v25 = (v23 - a2[12]) >> 3;
    v9[66] = a1[3];
    sub_2FB05A0(v9 + 67, v24, v25, v26, v27, v28);
    v9[86] = 0;
    v9[87] = 0;
    v9[88] = 0;
    *((_DWORD *)v9 + 178) = 0;
    v9[90] = 0;
    v9[91] = 0;
    v9[92] = 0;
    *((_DWORD *)v9 + 186) = 0;
    v9[94] = v9 + 96;
    v9[95] = 0;
    v9[96] = 0;
    v9[97] = 0;
    v9[98] = 0;
    *((_DWORD *)v9 + 198) = 0;
    v9[100] = a4;
  }
  return v9;
}
