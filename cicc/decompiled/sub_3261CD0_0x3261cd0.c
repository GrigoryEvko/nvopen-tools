// Function: sub_3261CD0
// Address: 0x3261cd0
//
__int64 __fastcall sub_3261CD0(__int64 *a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 v4; // r11
  int v5; // r10d
  __int64 v6; // r9
  int v7; // r8d
  __int64 v8; // rcx
  unsigned __int16 *v9; // rax
  __int64 v10; // rsi
  int v11; // r14d
  __int64 v12; // r15
  int v13; // eax
  __int64 v14; // rdi
  int v15; // ecx
  __int64 v16; // r14
  __int64 v17; // rax
  int v19; // [rsp+Ch] [rbp-94h]
  __int64 v20; // [rsp+10h] [rbp-90h]
  __int64 v21; // [rsp+18h] [rbp-88h]
  __int64 v22; // [rsp+20h] [rbp-80h]
  int v23; // [rsp+28h] [rbp-78h]
  int v24; // [rsp+2Ch] [rbp-74h]
  __int64 v25; // [rsp+30h] [rbp-70h] BYREF
  int v26; // [rsp+38h] [rbp-68h]
  __int64 v27; // [rsp+40h] [rbp-60h] BYREF
  int v28; // [rsp+48h] [rbp-58h]
  __int64 v29; // [rsp+50h] [rbp-50h]
  int v30; // [rsp+58h] [rbp-48h]
  __int64 v31; // [rsp+60h] [rbp-40h]
  int v32; // [rsp+68h] [rbp-38h]

  v3 = *(__int64 **)(a2 + 40);
  v4 = *v3;
  v5 = *((_DWORD *)v3 + 2);
  v6 = v3[5];
  v7 = *((_DWORD *)v3 + 12);
  v8 = v3[10];
  v24 = *((_DWORD *)v3 + 22);
  v9 = *(unsigned __int16 **)(a2 + 48);
  v10 = *(_QWORD *)(a2 + 80);
  v11 = *v9;
  v12 = *((_QWORD *)v9 + 1);
  v25 = v10;
  if ( v10 )
  {
    v23 = v5;
    v19 = v7;
    v20 = v8;
    v21 = v6;
    v22 = v4;
    sub_B96E90((__int64)&v25, v10, 1);
    v7 = v19;
    v5 = v23;
    v8 = v20;
    v6 = v21;
    v4 = v22;
  }
  v13 = *(_DWORD *)(a2 + 72);
  v14 = *a1;
  v29 = v6;
  v26 = v13;
  v30 = v7;
  v32 = v24;
  v31 = v8;
  v15 = v11;
  v16 = 0;
  v27 = v4;
  v28 = v5;
  v17 = sub_3402EA0(v14, 151, (unsigned int)&v25, v15, v12, 0, (__int64)&v27, 3);
  if ( v17 )
    v16 = v17;
  if ( v25 )
    sub_B91220((__int64)&v25, v25);
  return v16;
}
