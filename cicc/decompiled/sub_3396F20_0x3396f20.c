// Function: sub_3396F20
// Address: 0x3396f20
//
void __fastcall sub_3396F20(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v9; // rax
  int v10; // eax
  int v11; // edx
  int v12; // r9d
  int v13; // ecx
  int v14; // r8d
  int v15; // edx
  __int64 v16; // rax
  __int64 v17; // r10
  __int64 v18; // rsi
  __int64 v19; // r12
  int v20; // edx
  int v21; // r13d
  _QWORD *v22; // rax
  __int64 v23; // rsi
  bool v24; // al
  int v25; // [rsp+0h] [rbp-80h]
  int v26; // [rsp+8h] [rbp-78h]
  int v27; // [rsp+10h] [rbp-70h]
  int v28; // [rsp+10h] [rbp-70h]
  __int64 *v29; // [rsp+18h] [rbp-68h]
  int v30; // [rsp+18h] [rbp-68h]
  int v31; // [rsp+18h] [rbp-68h]
  __int64 v32; // [rsp+38h] [rbp-48h] BYREF
  __int64 v33; // [rsp+40h] [rbp-40h] BYREF
  int v34; // [rsp+48h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(__int64 **)(a2 - 8);
  else
    v3 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v4 = sub_338B750(a1, *v3);
  v5 = *(_QWORD *)(a1 + 864);
  v7 = v6;
  v8 = *(_QWORD *)(v5 + 16);
  v29 = *(__int64 **)(a2 + 8);
  v9 = sub_2E79000(*(__int64 **)(v5 + 40));
  v10 = sub_2D5BAE0(v8, v9, v29, 0);
  v12 = 0;
  v13 = v10;
  v14 = v11;
  if ( *(_BYTE *)a2 > 0x1Cu && ((*(_BYTE *)a2 - 68) & 0xFB) == 0 )
  {
    v28 = v10;
    v31 = v11;
    v24 = sub_B44910(a2);
    v13 = v28;
    v14 = v31;
    v12 = 16 * v24;
  }
  v15 = *(_DWORD *)(a1 + 848);
  v16 = *(_QWORD *)a1;
  v33 = 0;
  v17 = *(_QWORD *)(a1 + 864);
  v34 = v15;
  if ( v16 )
  {
    if ( &v33 != (__int64 *)(v16 + 48) )
    {
      v18 = *(_QWORD *)(v16 + 48);
      v33 = v18;
      if ( v18 )
      {
        v25 = v13;
        v26 = v14;
        v27 = v12;
        v30 = v17;
        sub_B96E90((__int64)&v33, v18, 1);
        v13 = v25;
        v14 = v26;
        v12 = v27;
        LODWORD(v17) = v30;
      }
    }
  }
  v32 = a2;
  v19 = sub_33FA050(v17, 221, (unsigned int)&v33, v13, v14, v12, v4, v7);
  v21 = v20;
  v22 = sub_337DC20(a1 + 8, &v32);
  *v22 = v19;
  v23 = v33;
  *((_DWORD *)v22 + 2) = v21;
  if ( v23 )
    sub_B91220((__int64)&v33, v23);
}
