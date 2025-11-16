// Function: sub_206F670
// Address: 0x206f670
//
__int64 *__fastcall sub_206F670(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v6; // r13
  __int64 **v7; // rax
  __int64 *v8; // rax
  __int64 v9; // r15
  __int64 v10; // rdx
  __int64 v11; // r14
  __int64 v12; // rax
  unsigned int v13; // eax
  unsigned int v14; // ecx
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rax
  unsigned int v18; // r8d
  __int64 v19; // rax
  unsigned int v20; // r8d
  unsigned int v21; // r9d
  __int64 (*v22)(); // rax
  int v23; // edx
  __int64 v24; // rax
  _QWORD *v25; // r11
  __int64 v26; // rsi
  int v27; // edx
  __int64 *result; // rax
  char v29; // al
  unsigned int v30; // [rsp+0h] [rbp-90h]
  unsigned int v31; // [rsp+Ch] [rbp-84h]
  unsigned int v32; // [rsp+10h] [rbp-80h]
  unsigned int v33; // [rsp+10h] [rbp-80h]
  _QWORD *v34; // [rsp+18h] [rbp-78h]
  unsigned int v35; // [rsp+18h] [rbp-78h]
  __int64 *v36; // [rsp+20h] [rbp-70h]
  unsigned int v37; // [rsp+20h] [rbp-70h]
  __int64 *v38; // [rsp+28h] [rbp-68h]
  __int64 v39; // [rsp+50h] [rbp-40h] BYREF
  int v40; // [rsp+58h] [rbp-38h]

  v6 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 16LL);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v7 = *(__int64 ***)(a2 - 8);
  else
    v7 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v36 = *v7;
  v8 = sub_20685E0(a1, *v7, a3, a4, a5);
  v9 = *(_QWORD *)a2;
  v38 = v8;
  v11 = v10;
  v12 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL));
  LOBYTE(v13) = sub_204D4D0(v6, v12, v9);
  v14 = v13;
  v16 = v15;
  v17 = *v36;
  if ( *(_BYTE *)(*v36 + 8) == 16 )
    v17 = **(_QWORD **)(v17 + 16);
  v18 = *(_DWORD *)(v17 + 8);
  v19 = *(_QWORD *)a2;
  v20 = v18 >> 8;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
    v19 = **(_QWORD **)(v19 + 16);
  v21 = *(_DWORD *)(v19 + 8) >> 8;
  v22 = *(__int64 (**)())(*(_QWORD *)v6 + 576LL);
  if ( v22 == sub_1D12D90
    || (v33 = v14,
        v35 = v21,
        v37 = v20,
        v29 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v22)(v6, v20, v21),
        v20 = v37,
        v21 = v35,
        v14 = v33,
        !v29) )
  {
    v23 = *(_DWORD *)(a1 + 536);
    v24 = *(_QWORD *)a1;
    v39 = 0;
    v25 = *(_QWORD **)(a1 + 552);
    v40 = v23;
    if ( v24 )
    {
      if ( &v39 != (__int64 *)(v24 + 48) )
      {
        v26 = *(_QWORD *)(v24 + 48);
        v39 = v26;
        if ( v26 )
        {
          v32 = v21;
          v30 = v14;
          v31 = v20;
          v34 = v25;
          sub_1623A60((__int64)&v39, v26, 2);
          v14 = v30;
          v20 = v31;
          v21 = v32;
          v25 = v34;
        }
      }
    }
    v38 = sub_1D2B130(v25, (__int64)&v39, v14, v16, (__int64)v38, v11, v20, v21);
    LODWORD(v11) = v27;
    if ( v39 )
      sub_161E7C0((__int64)&v39, v39);
  }
  v39 = a2;
  result = sub_205F5C0(a1 + 8, &v39);
  result[1] = (__int64)v38;
  *((_DWORD *)result + 4) = v11;
  return result;
}
