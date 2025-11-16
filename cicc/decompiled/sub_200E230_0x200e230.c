// Function: sub_200E230
// Address: 0x200e230
//
__int64 __fastcall sub_200E230(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        double a6,
        double a7,
        double a8)
{
  _QWORD *v8; // r9
  __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // r10
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // r9
  const void **v20; // rdx
  const void **v21; // r8
  __int64 v22; // rcx
  _DWORD *v23; // rdx
  bool v24; // bl
  int v25; // eax
  __int64 v26; // r12
  bool v28; // al
  __int128 v29; // [rsp-10h] [rbp-80h]
  _QWORD *v30; // [rsp+0h] [rbp-70h]
  __int64 v31; // [rsp+0h] [rbp-70h]
  __int64 v32; // [rsp+8h] [rbp-68h]
  const void **v33; // [rsp+8h] [rbp-68h]
  __int64 (__fastcall *v34)(__int64, __int64, __int64, _QWORD, __int64); // [rsp+10h] [rbp-60h]
  _QWORD *v35; // [rsp+10h] [rbp-60h]
  __int64 v36; // [rsp+18h] [rbp-58h]
  _DWORD *v37; // [rsp+18h] [rbp-58h]
  __int64 v38; // [rsp+20h] [rbp-50h] BYREF
  int v39; // [rsp+28h] [rbp-48h]
  _QWORD v40[8]; // [rsp+30h] [rbp-40h] BYREF

  v8 = a1;
  v13 = a2;
  v14 = *(_QWORD *)(a2 + 72);
  v38 = v14;
  if ( v14 )
  {
    sub_1623A60((__int64)&v38, v14, 2);
    v8 = a1;
    v13 = a2;
  }
  v15 = *v8;
  v30 = v8;
  v39 = *(_DWORD *)(v13 + 64);
  v16 = v8[1];
  v32 = v15;
  v34 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)v15 + 264LL);
  v36 = *(_QWORD *)(v16 + 48);
  v17 = sub_1E0A0C0(*(_QWORD *)(v16 + 32));
  v18 = v34(v32, v17, v36, (unsigned int)a4, a5);
  v19 = v30;
  v40[0] = a4;
  v21 = v20;
  v22 = v18;
  v40[1] = a5;
  v23 = (_DWORD *)*v30;
  if ( (_BYTE)a4 )
  {
    if ( (unsigned __int8)(a4 - 14) > 0x5Fu )
    {
      v24 = (unsigned __int8)(a4 - 86) <= 0x17u || (unsigned __int8)(a4 - 8) <= 5u;
      goto LABEL_6;
    }
LABEL_12:
    v25 = v23[17];
    goto LABEL_8;
  }
  v31 = v18;
  v33 = v21;
  v35 = v19;
  v37 = v23;
  v24 = sub_1F58CD0((__int64)v40);
  v28 = sub_1F58D20((__int64)v40);
  v23 = v37;
  v19 = v35;
  v21 = v33;
  v22 = v31;
  if ( v28 )
    goto LABEL_12;
LABEL_6:
  if ( v24 )
    v25 = v23[16];
  else
    v25 = v23[15];
LABEL_8:
  *((_QWORD *)&v29 + 1) = a3;
  *(_QWORD *)&v29 = a2;
  v26 = sub_1D309E0((__int64 *)v19[1], (unsigned int)(144 - v25), (__int64)&v38, v22, v21, 0, a6, a7, a8, v29);
  if ( v38 )
    sub_161E7C0((__int64)&v38, v38);
  return v26;
}
