// Function: sub_22C3280
// Address: 0x22c3280
//
__int64 __fastcall sub_22C3280(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  int v3; // eax
  int v4; // ecx
  unsigned int v5; // edx
  int v6; // eax
  int v7; // eax
  __int64 v8; // rax
  int v9; // eax
  unsigned __int64 *v10; // r15
  _QWORD *v11; // r13
  __int64 v12; // r14
  unsigned __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 result; // rax
  _QWORD *v16; // rbx
  _QWORD *v17; // r14
  _QWORD *v18; // r12
  __int64 i; // rcx
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // r8
  bool v23; // dl
  _QWORD *v24; // rdi
  _QWORD *v25; // rdi
  __int64 v26; // rax
  int v27; // edx
  int v28; // [rsp+Ch] [rbp-94h]
  _QWORD v29[2]; // [rsp+10h] [rbp-90h] BYREF
  __int64 v30; // [rsp+20h] [rbp-80h]
  _QWORD v31[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v32; // [rsp+40h] [rbp-60h]
  unsigned __int64 v33; // [rsp+50h] [rbp-50h] BYREF
  __int64 v34; // [rsp+58h] [rbp-48h]
  __int64 v35; // [rsp+60h] [rbp-40h]

  v2 = a1;
  v3 = *(_DWORD *)(a2 + 8);
  v4 = *(_DWORD *)(a1 + 8);
  v29[0] = 0;
  v29[1] = 0;
  v30 = -4096;
  v31[0] = 0;
  v31[1] = 0;
  *(_DWORD *)(a2 + 8) = v4 & 0xFFFFFFFE | v3 & 1;
  v5 = v3 & 0xFFFFFFFE;
  v6 = *(_DWORD *)(a1 + 8);
  v32 = -8192;
  *(_DWORD *)(a1 + 8) = v5 | v6 & 1;
  v7 = *(_DWORD *)(a1 + 12);
  *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
  *(_DWORD *)(a2 + 12) = v7;
  if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
  {
    if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
    {
      v8 = a2;
      a2 = a1;
      v2 = v8;
      goto LABEL_4;
    }
    v26 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
    v27 = *(_DWORD *)(a2 + 24);
    *(_QWORD *)(a2 + 16) = v26;
    LODWORD(v26) = *(_DWORD *)(a1 + 24);
    *(_DWORD *)(a1 + 24) = v27;
    *(_DWORD *)(a2 + 24) = v26;
LABEL_30:
    sub_D68D70(v31);
    return sub_D68D70(v29);
  }
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v16 = (_QWORD *)(a1 + 16);
    v17 = (_QWORD *)(a2 + 16);
    v18 = (_QWORD *)(a2 + 64);
    for ( i = -4096; ; i = v30 )
    {
      v20 = v16[2];
      v21 = v17[2];
      v22 = v21;
      v23 = v20 != 0 && v20 != -4096 && v20 != -8192;
      if ( v20 == i || i == v21 || v32 == v21 || v32 == v20 )
      {
        v33 = 0;
        v34 = 0;
        v35 = v20;
        if ( v23 )
        {
          sub_BD6050(&v33, *v16 & 0xFFFFFFFFFFFFFFF8LL);
          v22 = v17[2];
        }
        v21 = v22;
      }
      else
      {
        v33 = 0;
        v34 = 0;
        v35 = v20;
        if ( v23 )
        {
          sub_BD6050(&v33, *v16 & 0xFFFFFFFFFFFFFFF8LL);
          v21 = v17[2];
        }
      }
      v24 = v16;
      v16 += 3;
      sub_22BDC40(v24, v21);
      v25 = v17;
      v17 += 3;
      sub_22BDC40(v25, v35);
      sub_D68D70(&v33);
      if ( v18 == v17 )
        break;
    }
    goto LABEL_30;
  }
LABEL_4:
  v9 = *(_DWORD *)(a2 + 24);
  *(_BYTE *)(a2 + 8) |= 1u;
  v10 = (unsigned __int64 *)(a2 + 16);
  v11 = (_QWORD *)(v2 + 16);
  v12 = *(_QWORD *)(a2 + 16);
  v28 = v9;
  do
  {
    *v10 = 0;
    v10[1] = 0;
    v13 = v11[2];
    v10[2] = v13;
    if ( v13 != -4096 && v13 != 0 && v13 != -8192 )
      sub_BD6050(v10, *v11 & 0xFFFFFFFFFFFFFFF8LL);
    v14 = v11[2];
    if ( v14 != 0 && v14 != -4096 && v14 != -8192 )
      sub_BD60C0(v11);
    v10 += 3;
    v11 += 3;
  }
  while ( (unsigned __int64 *)(a2 + 64) != v10 );
  *(_BYTE *)(v2 + 8) &= ~1u;
  *(_QWORD *)(v2 + 16) = v12;
  *(_DWORD *)(v2 + 24) = v28;
  if ( v32 != -4096 && v32 != 0 && v32 != -8192 )
    sub_BD60C0(v31);
  result = v30;
  if ( v30 != -4096 && v30 != 0 && v30 != -8192 )
    return sub_BD60C0(v29);
  return result;
}
