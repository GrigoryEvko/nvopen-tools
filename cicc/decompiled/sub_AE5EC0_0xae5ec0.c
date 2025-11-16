// Function: sub_AE5EC0
// Address: 0xae5ec0
//
__int64 __fastcall sub_AE5EC0(__int64 **a1)
{
  __int64 v1; // r15
  __int64 v2; // rax
  int v3; // r13d
  __int64 v4; // rbx
  unsigned __int8 v5; // dl
  __int64 *v6; // rsi
  __int64 v7; // r8
  __int64 v8; // rdx
  __int64 v9; // r9
  __int64 v10; // r12
  __int64 v11; // rax
  _QWORD *v12; // r14
  __int64 v13; // rdx
  int v14; // ecx
  __int64 v15; // rsi
  int v16; // eax
  int v17; // edx
  int v18; // eax
  __int64 v20; // [rsp+0h] [rbp-70h]
  __int64 v21; // [rsp+8h] [rbp-68h]
  int v22; // [rsp+8h] [rbp-68h]
  int v23; // [rsp+14h] [rbp-5Ch]
  int v24; // [rsp+18h] [rbp-58h]
  int v25; // [rsp+1Ch] [rbp-54h]
  int v26; // [rsp+20h] [rbp-50h]
  int v27; // [rsp+24h] [rbp-4Ch]
  __int64 v28; // [rsp+28h] [rbp-48h]
  __int64 v29; // [rsp+30h] [rbp-40h]
  __int64 v30; // [rsp+38h] [rbp-38h]

  v1 = *a1[7];
  v30 = *a1[5];
  v2 = **a1;
  v3 = *(_DWORD *)(v2 + 16);
  v23 = *(_DWORD *)(v2 + 36);
  v25 = *(_DWORD *)(v2 + 28);
  v24 = *(_DWORD *)(v2 + 32);
  v29 = *a1[4];
  v26 = *(_DWORD *)(v2 + 24);
  v28 = *a1[3];
  v27 = *(_DWORD *)(v2 + 20);
  v4 = *a1[1];
  v5 = *(_BYTE *)(v2 - 16);
  v6 = a1[2];
  if ( (v5 & 2) != 0 )
  {
    v7 = *(_QWORD *)(*(_QWORD *)(v2 - 32) + 16LL);
    if ( v7 )
    {
LABEL_3:
      v7 = sub_B91420(v7, v6);
      v9 = v8;
      v10 = *a1[1];
      v2 = **a1;
      goto LABEL_4;
    }
  }
  else
  {
    v7 = *(_QWORD *)(v2 - 8LL * ((v5 >> 2) & 0xF));
    if ( v7 )
      goto LABEL_3;
  }
  LODWORD(v10) = *a1[1];
  v9 = 0;
LABEL_4:
  v11 = *(_QWORD *)(v2 + 8);
  v12 = (_QWORD *)(v11 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v11 & 4) != 0 )
    v12 = (_QWORD *)*v12;
  v13 = v6[1];
  v14 = 0;
  v15 = *v6;
  if ( v13 )
  {
    v20 = v7;
    v21 = v9;
    v16 = sub_B9B140(v12, v15, v13);
    v7 = v20;
    v9 = v21;
    v14 = v16;
  }
  v17 = 0;
  if ( v9 )
  {
    v22 = v14;
    v18 = sub_B9B140(v12, v7, v9);
    v14 = v22;
    v17 = v18;
  }
  return sub_B07EA0((_DWORD)v12, v10, v17, v14, v4, v3, v28, v27, v29, v26, v25, v24, v23, v30, 0, v1, 0, 0, 0, 0, 1, 1);
}
