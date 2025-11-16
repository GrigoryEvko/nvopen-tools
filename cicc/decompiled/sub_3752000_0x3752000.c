// Function: sub_3752000
// Address: 0x3752000
//
__int64 __fastcall sub_3752000(__int64 *a1, unsigned __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v9; // r8
  __int64 v10; // rsi
  int v11; // r10d
  __int64 v12; // rax
  int v13; // r11d
  unsigned int i; // edx
  __int64 v15; // rcx
  unsigned int v16; // edx
  __int64 v18; // rcx
  __int64 v20; // rdi
  char v21; // cl
  __int64 v22; // rsi
  __int64 (__fastcall *v23)(__int64, unsigned __int16); // rax
  __int64 v24; // rsi
  unsigned __int32 v25; // eax
  unsigned __int8 *v26; // rsi
  unsigned __int32 v27; // r13d
  __int64 v28; // rax
  __int64 v29; // r14
  __int64 *v30; // rsi
  __int64 v31; // rdi
  unsigned __int8 *v32; // [rsp+8h] [rbp-58h] BYREF
  __int64 v33[10]; // [rsp+10h] [rbp-50h] BYREF

  if ( *(_DWORD *)(a2 + 24) == -11 )
  {
    v20 = a1[4];
    v21 = *(_BYTE *)(a2 + 32);
    v22 = *(unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3);
    v23 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v20 + 552LL);
    if ( v23 == sub_2EC09E0 )
      v24 = *(_QWORD *)(v20 + 8 * v22 + 112);
    else
      v24 = ((__int64 (__fastcall *)(__int64, __int64, bool))v23)(v20, v22, (v21 & 4) != 0);
    v25 = sub_2EC06C0(a1[1], v24, byte_3F871B3, 0, a5, a6);
    v26 = *(unsigned __int8 **)(a2 + 80);
    v27 = v25;
    v28 = a1[2];
    v32 = v26;
    v29 = *(_QWORD *)(v28 + 8) - 400LL;
    if ( v26 )
    {
      sub_B96E90((__int64)&v32, (__int64)v26, 1);
      v33[0] = (__int64)v32;
      if ( v32 )
      {
        sub_B976B0((__int64)&v32, v32, (__int64)v33);
        v32 = 0;
      }
    }
    else
    {
      v33[0] = 0;
    }
    v30 = (__int64 *)a1[6];
    v31 = a1[5];
    v33[1] = 0;
    v33[2] = 0;
    sub_2F26260(v31, v30, v33, v29, v27);
    if ( v33[0] )
      sub_B91220((__int64)v33, v33[0]);
    if ( v32 )
      sub_B91220((__int64)&v32, (__int64)v32);
    return v27;
  }
  v9 = *(_BYTE *)(a4 + 8) & 1;
  if ( v9 )
  {
    v10 = a4 + 16;
    v11 = 15;
  }
  else
  {
    v12 = *(unsigned int *)(a4 + 24);
    v10 = *(_QWORD *)(a4 + 16);
    if ( !(_DWORD)v12 )
      goto LABEL_13;
    v11 = v12 - 1;
  }
  v13 = 1;
  for ( i = v11 & (((a2 >> 9) ^ (a2 >> 4)) + a3); ; i = v11 & v16 )
  {
    v15 = v10 + 24LL * i;
    if ( a2 != *(_QWORD *)v15 )
      break;
    if ( *(_DWORD *)(v15 + 8) == a3 )
      return *(unsigned int *)(v15 + 16);
LABEL_9:
    v16 = v13 + i;
    ++v13;
  }
  if ( *(_QWORD *)v15 || *(_DWORD *)(v15 + 8) != -1 )
    goto LABEL_9;
  if ( v9 )
  {
    v18 = 384;
    goto LABEL_14;
  }
  v12 = *(unsigned int *)(a4 + 24);
LABEL_13:
  v18 = 24 * v12;
LABEL_14:
  v15 = v10 + v18;
  return *(unsigned int *)(v15 + 16);
}
