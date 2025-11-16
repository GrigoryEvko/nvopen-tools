// Function: sub_15AB030
// Address: 0x15ab030
//
__int64 __fastcall sub_15AB030(__int64 **a1)
{
  __int64 v1; // r14
  __int64 v2; // rax
  char v3; // dl
  int v4; // r12d
  __int64 *v5; // rcx
  __int64 *v6; // rsi
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r9
  __int64 v13; // r15
  __int64 v14; // rax
  _QWORD *v15; // r13
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // ecx
  __int64 v19; // rsi
  int v20; // eax
  int v21; // edx
  int v22; // eax
  __int64 v24; // [rsp+8h] [rbp-78h]
  __int64 *v25; // [rsp+10h] [rbp-70h]
  __int64 v26; // [rsp+10h] [rbp-70h]
  int v27; // [rsp+10h] [rbp-70h]
  _BOOL4 v28; // [rsp+18h] [rbp-68h]
  _BOOL4 v29; // [rsp+1Ch] [rbp-64h]
  __int64 *v30; // [rsp+20h] [rbp-60h]
  __int64 v31; // [rsp+20h] [rbp-60h]
  _BOOL4 v32; // [rsp+28h] [rbp-58h]
  int v33; // [rsp+2Ch] [rbp-54h]
  int v34; // [rsp+30h] [rbp-50h]
  int v35; // [rsp+34h] [rbp-4Ch]
  int v36; // [rsp+38h] [rbp-48h]
  int v37; // [rsp+3Ch] [rbp-44h]
  __int64 v38; // [rsp+40h] [rbp-40h]
  __int64 v39; // [rsp+48h] [rbp-38h]

  v1 = *a1[7];
  v39 = *a1[5];
  v2 = **a1;
  v3 = *(_BYTE *)(v2 + 40);
  v4 = *(_DWORD *)(v2 + 24);
  v34 = *(_DWORD *)(v2 + 36);
  v37 = *(_DWORD *)(v2 + 28);
  v5 = a1[4];
  v32 = (v3 & 0x10) != 0;
  v33 = *(_DWORD *)(v2 + 44);
  v35 = *(_DWORD *)(v2 + 32);
  v36 = v3 & 3;
  v28 = (v3 & 4) != 0;
  v29 = (v3 & 8) != 0;
  v38 = *a1[3];
  v6 = a1[2];
  v7 = *a1[1];
  v8 = 2LL - *(unsigned int *)(v2 + 8);
  v9 = *(_QWORD *)(v2 + 8 * v8);
  if ( v9 )
  {
    v25 = a1[2];
    v30 = a1[4];
    v10 = sub_161E970(*(_QWORD *)(v2 + 8 * v8));
    v5 = v30;
    v6 = v25;
    v9 = v10;
    v12 = v11;
    v13 = *a1[1];
    v2 = **a1;
  }
  else
  {
    v13 = *a1[1];
    v12 = 0;
  }
  v14 = *(_QWORD *)(v2 + 16);
  v15 = (_QWORD *)(v14 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v14 & 4) != 0 )
    v15 = (_QWORD *)*v15;
  v16 = *v5;
  v17 = v6[1];
  v18 = 0;
  v19 = *v6;
  v31 = v16;
  if ( v17 )
  {
    v24 = v12;
    v26 = v9;
    v20 = sub_161FF10(v15, v19, v17);
    v12 = v24;
    v9 = v26;
    v18 = v20;
  }
  v21 = 0;
  if ( v12 )
  {
    v27 = v18;
    v22 = sub_161FF10(v15, v9, v12);
    v18 = v27;
    v21 = v22;
  }
  return sub_15BFC70(
           (_DWORD)v15,
           v13,
           v21,
           v18,
           v7,
           v4,
           v38,
           v28,
           v29,
           v37,
           v31,
           v36,
           v35,
           v34,
           v33,
           v32,
           v39,
           0,
           v1,
           0,
           0,
           1,
           1);
}
