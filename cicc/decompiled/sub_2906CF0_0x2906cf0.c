// Function: sub_2906CF0
// Address: 0x2906cf0
//
char __fastcall sub_2906CF0(__int64 **a1, __int64 *a2)
{
  __int64 *v2; // rbx
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdi
  unsigned int v9; // r10d
  __int64 *v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rsi
  __int64 v13; // rax
  char result; // al
  __int64 v15; // r13
  char v16; // dl
  char v17; // r13
  int v18; // r9d
  __int64 v19; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v20; // [rsp+18h] [rbp-A8h]
  __int64 v21; // [rsp+20h] [rbp-A0h]
  int v22; // [rsp+28h] [rbp-98h]
  _QWORD v23[4]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v24; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v25[4]; // [rsp+58h] [rbp-68h] BYREF
  _QWORD v26[9]; // [rsp+78h] [rbp-48h] BYREF

  v2 = *a1;
  v3 = *a2;
  v4 = sub_2906530(*a2, **a1, (*a1)[1]);
  v5 = v2[1];
  v6 = v4;
  v7 = *(unsigned int *)(v5 + 24);
  v8 = *(_QWORD *)(v5 + 8);
  if ( !(_DWORD)v7 )
  {
LABEL_18:
    v12 = *(_QWORD *)(v5 + 32);
    goto LABEL_19;
  }
  v9 = (v7 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v10 = (__int64 *)(v8 + 16LL * v9);
  v11 = *v10;
  if ( v6 != *v10 )
  {
    v18 = 1;
    while ( v11 != -4096 )
    {
      v9 = (v7 - 1) & (v9 + v18);
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( v6 == *v10 )
        goto LABEL_3;
      ++v18;
    }
    goto LABEL_18;
  }
LABEL_3:
  v12 = *(_QWORD *)(v5 + 32);
  if ( v10 != (__int64 *)(v8 + 16 * v7) )
  {
    v13 = v12 + 16LL * *((unsigned int *)v10 + 2);
    goto LABEL_5;
  }
LABEL_19:
  v13 = v12 + 16LL * *(unsigned int *)(v5 + 40);
LABEL_5:
  if ( *(_BYTE *)(v13 + 8) )
  {
    result = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v3 + 8) + 8LL) - 17 <= 1;
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v6 + 8) + 8LL) - 17 <= 1 == result )
      return result;
    v19 = 0;
    v15 = v2[2];
    v20 = 0;
    v21 = v6;
    goto LABEL_8;
  }
  v15 = v2[2];
  v19 = 0;
  v20 = 0;
  v21 = v6;
  if ( v6 )
  {
LABEL_8:
    if ( v6 != -4096 && v6 != -8192 )
      sub_BD73F0((__int64)&v19);
  }
  v24 = v6;
  v22 = 0;
  memset(v23, 0, 24);
  sub_28FF950((__int64)v25, (__int64)&v19);
  sub_2906B20(v15, &v24, (__int64)v25);
  v17 = v16;
  sub_D68D70(v26);
  sub_D68D70(v25);
  sub_D68D70(v23);
  result = sub_D68D70(&v19);
  if ( v17 )
    return sub_94F890(v2[3], v6);
  return result;
}
