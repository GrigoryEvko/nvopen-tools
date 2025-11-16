// Function: sub_2908C80
// Address: 0x2908c80
//
__int64 __fastcall sub_2908C80(__int64 a1, _QWORD *a2)
{
  __int64 v4; // rax
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // r12
  unsigned int v9; // esi
  int v10; // eax
  __int64 v11; // rbx
  int v12; // eax
  __int64 v13; // rcx
  __int64 v14; // rsi
  unsigned __int64 v15; // rdx
  __int64 v16; // r9
  int v17; // eax
  __int64 v18; // rdi
  unsigned __int64 *v19; // rdx
  unsigned __int64 *v20; // r12
  unsigned __int64 v21; // r12
  __int64 v22; // rdi
  __int64 v23; // rax
  unsigned __int64 v24; // rsi
  unsigned __int64 *v25; // [rsp+10h] [rbp-C0h]
  __int64 v26; // [rsp+28h] [rbp-A8h] BYREF
  _QWORD v27[4]; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int64 v28; // [rsp+50h] [rbp-80h] BYREF
  __int64 v29; // [rsp+58h] [rbp-78h]
  __int64 v30; // [rsp+60h] [rbp-70h]
  int v31; // [rsp+68h] [rbp-68h]
  unsigned __int64 v32; // [rsp+70h] [rbp-60h] BYREF
  __int64 v33; // [rsp+78h] [rbp-58h]
  __int64 v34; // [rsp+80h] [rbp-50h]
  unsigned __int64 v35[9]; // [rsp+88h] [rbp-48h] BYREF

  v4 = a2[2];
  v32 = 0;
  v33 = 0;
  v34 = v4;
  if ( v4 == 0 || v4 == -4096 || v4 == -8192 )
  {
    LODWORD(v35[0]) = 0;
    v28 = 0;
    v29 = 0;
    v30 = v4;
    v5 = 0;
  }
  else
  {
    sub_BD6050(&v32, *a2 & 0xFFFFFFFFFFFFFFF8LL);
    LODWORD(v35[0]) = 0;
    v28 = 0;
    v29 = 0;
    v30 = v34;
    if ( v34 == -8192 || v34 == 0 || v34 == -4096 )
    {
      v5 = 0;
    }
    else
    {
      sub_BD6050(&v28, v32 & 0xFFFFFFFFFFFFFFF8LL);
      v5 = v35[0];
    }
  }
  v31 = v5;
  sub_D68D70(&v32);
  if ( (unsigned __int8)sub_2901500(a1, (__int64)&v28, &v26) )
  {
    v6 = *(unsigned int *)(v26 + 24);
    goto LABEL_9;
  }
  v9 = *(_DWORD *)(a1 + 24);
  v10 = *(_DWORD *)(a1 + 16);
  v11 = v26;
  ++*(_QWORD *)a1;
  v12 = v10 + 1;
  v27[0] = v11;
  if ( 4 * v12 >= 3 * v9 )
  {
    sub_2908890(a1, 2 * v9);
  }
  else
  {
    if ( v9 - *(_DWORD *)(a1 + 20) - v12 > v9 >> 3 )
      goto LABEL_12;
    sub_2908890(a1, v9);
  }
  sub_2901500(a1, (__int64)&v28, v27);
  v11 = v27[0];
  v12 = *(_DWORD *)(a1 + 16) + 1;
LABEL_12:
  *(_DWORD *)(a1 + 16) = v12;
  v32 = 0;
  v33 = 0;
  v34 = -4096;
  if ( *(_QWORD *)(v11 + 16) != -4096 )
    --*(_DWORD *)(a1 + 20);
  sub_D68D70(&v32);
  sub_FC7530((_QWORD *)v11, v30);
  *(_DWORD *)(v11 + 24) = v31;
  memset(v27, 0, 24);
  sub_D68CD0(&v32, 0, a2);
  sub_D68CD0(v35, 0, v27);
  v14 = *(unsigned int *)(a1 + 40);
  v15 = *(unsigned int *)(a1 + 44);
  v16 = v14 + 1;
  v17 = *(_DWORD *)(a1 + 40);
  if ( v14 + 1 > v15 )
  {
    v21 = *(_QWORD *)(a1 + 32);
    v22 = a1 + 32;
    if ( v21 > (unsigned __int64)&v32 )
    {
      v24 = v14 + 1;
    }
    else
    {
      v23 = 3 * v14;
      v24 = v14 + 1;
      if ( (unsigned __int64)&v32 < v21 + 16 * v23 )
      {
        sub_2903DC0(v22, v24, v15, v13, (__int64)v35, v16);
        v18 = *(_QWORD *)(a1 + 32);
        v14 = *(unsigned int *)(a1 + 40);
        v19 = (unsigned __int64 *)((char *)&v32 + v18 - v21);
        v17 = *(_DWORD *)(a1 + 40);
        goto LABEL_16;
      }
    }
    sub_2903DC0(v22, v24, v15, v13, (__int64)v35, v16);
    v14 = *(unsigned int *)(a1 + 40);
    v18 = *(_QWORD *)(a1 + 32);
    v19 = &v32;
    v17 = *(_DWORD *)(a1 + 40);
    goto LABEL_16;
  }
  v18 = *(_QWORD *)(a1 + 32);
  v19 = &v32;
LABEL_16:
  v20 = (unsigned __int64 *)(v18 + 48 * v14);
  if ( v20 )
  {
    v25 = v19;
    sub_D68CD0(v20, 0, v19);
    sub_D68CD0(v20 + 3, 0, v25 + 3);
    v17 = *(_DWORD *)(a1 + 40);
  }
  *(_DWORD *)(a1 + 40) = v17 + 1;
  sub_D68D70(v35);
  sub_D68D70(&v32);
  sub_D68D70(v27);
  v6 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  *(_DWORD *)(v11 + 24) = v6;
LABEL_9:
  v7 = *(_QWORD *)(a1 + 32) + 48 * v6;
  sub_D68D70(&v28);
  return v7 + 24;
}
