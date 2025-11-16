// Function: sub_3397460
// Address: 0x3397460
//
_QWORD *__fastcall sub_3397460(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r12
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r11
  __int64 v15; // r9
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // r15
  unsigned int v20; // edx
  unsigned __int64 v21; // r12
  int v22; // edx
  __int64 v23; // rax
  __int64 v24; // r11
  __int64 v25; // rsi
  __int64 v26; // r15
  int v27; // edx
  int v28; // r12d
  _QWORD *result; // rax
  __int64 v30; // [rsp+8h] [rbp-98h]
  __int64 v31; // [rsp+10h] [rbp-90h]
  __int64 v32; // [rsp+18h] [rbp-88h]
  __int64 v33; // [rsp+18h] [rbp-88h]
  __int64 v34; // [rsp+18h] [rbp-88h]
  unsigned int v35; // [rsp+20h] [rbp-80h]
  __int64 *v36; // [rsp+28h] [rbp-78h]
  __int64 v37; // [rsp+28h] [rbp-78h]
  __int64 v38; // [rsp+60h] [rbp-40h] BYREF
  int v39; // [rsp+68h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(__int64 **)(a2 - 8);
  else
    v3 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v4 = sub_338B750(a1, *v3);
  v5 = *(_QWORD *)(a1 + 864);
  v7 = v6;
  v8 = *(_QWORD *)(v5 + 16);
  v36 = *(__int64 **)(a2 + 8);
  v9 = sub_2E79000(*(__int64 **)(v5 + 40));
  v35 = sub_2D5BAE0(v8, v9, v36, 0);
  v37 = v10;
  v32 = *(_QWORD *)(a2 + 8);
  v11 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
  v12 = sub_336EEB0(v8, v11, v32, 0);
  v38 = 0;
  v14 = *(_QWORD *)(a1 + 864);
  v15 = v13;
  v16 = v12;
  v17 = *(_QWORD *)a1;
  v39 = *(_DWORD *)(a1 + 848);
  if ( v17 )
  {
    if ( &v38 != (__int64 *)(v17 + 48) )
    {
      v18 = *(_QWORD *)(v17 + 48);
      v38 = v18;
      if ( v18 )
      {
        v30 = v16;
        v31 = v13;
        v33 = v14;
        sub_B96E90((__int64)&v38, v18, 1);
        v16 = v30;
        v15 = v31;
        v14 = v33;
      }
    }
  }
  v19 = sub_33FB310(v14, v4, v7, &v38, v16, v15);
  v21 = v20 | v7 & 0xFFFFFFFF00000000LL;
  if ( v38 )
    sub_B91220((__int64)&v38, v38);
  v22 = *(_DWORD *)(a1 + 848);
  v23 = *(_QWORD *)a1;
  v38 = 0;
  v24 = *(_QWORD *)(a1 + 864);
  v39 = v22;
  if ( v23 )
  {
    if ( &v38 != (__int64 *)(v23 + 48) )
    {
      v25 = *(_QWORD *)(v23 + 48);
      v38 = v25;
      if ( v25 )
      {
        v34 = v24;
        sub_B96E90((__int64)&v38, v25, 1);
        v24 = v34;
      }
    }
  }
  v26 = sub_33FB4C0(v24, v19, v21, &v38, v35, v37);
  v28 = v27;
  if ( v38 )
    sub_B91220((__int64)&v38, v38);
  v38 = a2;
  result = sub_337DC20(a1 + 8, &v38);
  *result = v26;
  *((_DWORD *)result + 2) = v28;
  return result;
}
