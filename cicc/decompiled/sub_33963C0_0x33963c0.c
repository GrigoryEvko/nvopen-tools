// Function: sub_33963C0
// Address: 0x33963c0
//
void __fastcall sub_33963C0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v9; // rax
  int v10; // eax
  int v11; // r9d
  int v12; // edx
  int v13; // r8d
  int v14; // ecx
  int v15; // edx
  __int64 v16; // rax
  __int64 v17; // r10
  __int64 v18; // rsi
  __int64 v19; // r12
  int v20; // edx
  int v21; // r13d
  _QWORD *v22; // rax
  __int64 v23; // rsi
  int v24; // [rsp+0h] [rbp-80h]
  int v25; // [rsp+8h] [rbp-78h]
  int v26; // [rsp+14h] [rbp-6Ch]
  __int64 *v27; // [rsp+18h] [rbp-68h]
  int v28; // [rsp+18h] [rbp-68h]
  __int64 v29; // [rsp+38h] [rbp-48h] BYREF
  __int64 v30; // [rsp+40h] [rbp-40h] BYREF
  int v31; // [rsp+48h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(__int64 **)(a2 - 8);
  else
    v3 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v4 = sub_338B750(a1, *v3);
  v5 = *(_QWORD *)(a1 + 864);
  v7 = v6;
  v8 = *(_QWORD *)(v5 + 16);
  v27 = *(__int64 **)(a2 + 8);
  v9 = sub_2E79000(*(__int64 **)(v5 + 40));
  v10 = sub_2D5BAE0(v8, v9, v27, 0);
  v11 = 0;
  v13 = v12;
  v14 = v10;
  if ( *(_BYTE *)a2 == 67 )
    v11 = (*(_BYTE *)(a2 + 1) >> 1) & 3;
  v15 = *(_DWORD *)(a1 + 848);
  v16 = *(_QWORD *)a1;
  v30 = 0;
  v17 = *(_QWORD *)(a1 + 864);
  v31 = v15;
  if ( v16 )
  {
    if ( &v30 != (__int64 *)(v16 + 48) )
    {
      v18 = *(_QWORD *)(v16 + 48);
      v30 = v18;
      if ( v18 )
      {
        v24 = v14;
        v25 = v13;
        v26 = v11;
        v28 = v17;
        sub_B96E90((__int64)&v30, v18, 1);
        v14 = v24;
        v13 = v25;
        v11 = v26;
        LODWORD(v17) = v28;
      }
    }
  }
  v29 = a2;
  v19 = sub_33FA050(v17, 216, (unsigned int)&v30, v14, v13, v11, v4, v7);
  v21 = v20;
  v22 = sub_337DC20(a1 + 8, &v29);
  *v22 = v19;
  v23 = v30;
  *((_DWORD *)v22 + 2) = v21;
  if ( v23 )
    sub_B91220((__int64)&v30, v23);
}
