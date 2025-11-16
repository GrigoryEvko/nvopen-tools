// Function: sub_94C360
// Address: 0x94c360
//
__int64 __fastcall sub_94C360(__int64 a1, __int64 a2, int a3, unsigned __int64 *a4)
{
  __int64 v4; // rdx
  unsigned int v6; // r13d
  __int64 v9; // rsi
  __m128i *v10; // rax
  __int64 *v11; // rdi
  __int64 v12; // rax
  unsigned __int64 v13; // rsi
  __int64 v14; // r13
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // r11
  __int64 v18; // rdi
  __int64 (__fastcall *v19)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v20; // rax
  __int64 v21; // r15
  __int64 v23; // rax
  unsigned int *v24; // rax
  unsigned int *v25; // r13
  unsigned int *v26; // rbx
  __int64 v27; // rdx
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // [rsp+8h] [rbp-A8h]
  __int64 v31; // [rsp+8h] [rbp-A8h]
  __int64 v32; // [rsp+8h] [rbp-A8h]
  _QWORD v33[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v34; // [rsp+40h] [rbp-70h]
  _BYTE v35[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v36; // [rsp+70h] [rbp-40h]

  v4 = (unsigned int)(a3 - 189);
  v6 = 8928;
  v9 = *(_QWORD *)(a4[9] + 16);
  if ( (unsigned int)v4 <= 0xD )
    v6 = dword_3F14740[v4];
  v36 = 257;
  v10 = sub_92F410(a2, v9);
  v11 = *(__int64 **)(a2 + 32);
  v33[0] = v10;
  v12 = sub_90A810(v11, v6, 0, 0);
  v13 = 0;
  if ( v12 )
    v13 = *(_QWORD *)(v12 + 24);
  v14 = sub_921880((unsigned int **)(a2 + 48), v13, v12, (int)v33, 1, (__int64)v35, 0);
  v16 = sub_91A390(*(_QWORD *)(a2 + 32) + 8LL, *a4, 0, v15);
  v34 = 257;
  v17 = v16;
  if ( v16 == *(_QWORD *)(v14 + 8) )
  {
    v21 = v14;
    goto LABEL_12;
  }
  v18 = *(_QWORD *)(a2 + 128);
  v19 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v18 + 120LL);
  if ( v19 != sub_920130 )
  {
    v32 = v17;
    v29 = v19(v18, 39u, (_BYTE *)v14, v17);
    v17 = v32;
    v21 = v29;
    goto LABEL_11;
  }
  if ( *(_BYTE *)v14 <= 0x15u )
  {
    v30 = v17;
    if ( (unsigned __int8)sub_AC4810(39) )
      v20 = sub_ADAB70(39, v14, v30, 0);
    else
      v20 = sub_AA93C0(39, v14, v30);
    v17 = v30;
    v21 = v20;
LABEL_11:
    if ( v21 )
      goto LABEL_12;
  }
  v31 = v17;
  v36 = 257;
  v23 = sub_BD2C40(72, unk_3F10A14);
  v21 = v23;
  if ( v23 )
    sub_B515B0(v23, v14, v31, v35, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
    *(_QWORD *)(a2 + 136),
    v21,
    v33,
    *(_QWORD *)(a2 + 104),
    *(_QWORD *)(a2 + 112));
  v24 = *(unsigned int **)(a2 + 48);
  v25 = &v24[4 * *(unsigned int *)(a2 + 56)];
  if ( v24 != v25 )
  {
    v26 = *(unsigned int **)(a2 + 48);
    do
    {
      v27 = *((_QWORD *)v26 + 1);
      v28 = *v26;
      v26 += 4;
      sub_B99FD0(v21, v28, v27);
    }
    while ( v25 != v26 );
  }
LABEL_12:
  *(_QWORD *)a1 = v21;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
