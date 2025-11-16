// Function: sub_A67E40
// Address: 0xa67e40
//
__int64 __fastcall sub_A67E40(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 result; // rax
  __int64 v5; // r14
  __int64 v6; // rdx
  __int64 v7; // rsi
  unsigned int v8; // ecx
  __int64 v9; // rbx
  __int64 v10; // rdi
  int v11; // r15d
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  bool v16; // al
  bool v17; // zf
  __int64 v18; // r14
  __int64 v19; // rax
  int v20; // r9d
  unsigned int v21; // esi
  int v22; // eax
  int v23; // eax
  __int64 v24; // r12
  __int64 v25; // [rsp+10h] [rbp-100h] BYREF
  __int64 v26; // [rsp+18h] [rbp-F8h] BYREF
  void *v27; // [rsp+20h] [rbp-F0h]
  _QWORD v28[2]; // [rsp+28h] [rbp-E8h] BYREF
  __int64 v29; // [rsp+38h] [rbp-D8h]
  __int64 v30; // [rsp+40h] [rbp-D0h]
  void *v31; // [rsp+50h] [rbp-C0h]
  _QWORD v32[2]; // [rsp+58h] [rbp-B8h] BYREF
  __int64 v33; // [rsp+68h] [rbp-A8h]
  __int64 v34; // [rsp+70h] [rbp-A0h]
  void *v35; // [rsp+80h] [rbp-90h]
  _QWORD v36[2]; // [rsp+88h] [rbp-88h] BYREF
  __int64 v37; // [rsp+98h] [rbp-78h]
  __int64 v38; // [rsp+A0h] [rbp-70h]
  void *v39; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v40; // [rsp+B8h] [rbp-58h] BYREF
  __int64 v41; // [rsp+C0h] [rbp-50h]
  __int64 v42; // [rsp+C8h] [rbp-48h]
  __int64 v43; // [rsp+D0h] [rbp-40h]
  int v44; // [rsp+D8h] [rbp-38h]

  v3 = a1[1];
  v28[1] = 0;
  v28[0] = v3 & 6;
  v29 = a1[3];
  result = v29;
  if ( v29 != -4096 && v29 != 0 && v29 != -8192 )
  {
    sub_BD6050(v28, v3 & 0xFFFFFFFFFFFFFFF8LL);
    result = v29;
  }
  v5 = a1[4];
  v30 = v5;
  v27 = &unk_49D9A58;
  v6 = *(unsigned int *)(v5 + 24);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD *)(v5 + 8);
    v8 = (v6 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
    v9 = v7 + 48LL * v8;
    v10 = *(_QWORD *)(v9 + 24);
    if ( result == v10 )
    {
LABEL_6:
      if ( v9 == v7 + 48 * v6 )
        goto LABEL_30;
      v11 = *(_DWORD *)(v9 + 40);
      v40 = 2;
      v41 = 0;
      v42 = -8192;
      v39 = &unk_49D9A58;
      v43 = 0;
      v12 = *(_QWORD *)(v9 + 24);
      if ( v12 == -8192 )
      {
        *(_QWORD *)(v9 + 32) = 0;
      }
      else
      {
        if ( !v12 || v12 == -4096 )
        {
          *(_QWORD *)(v9 + 24) = -8192;
          v14 = v42;
          v15 = v43;
          v16 = v42 != 0;
          v17 = v42 == -4096;
        }
        else
        {
          sub_BD60C0(v9 + 8);
          v13 = v42;
          v17 = v42 == 0;
          *(_QWORD *)(v9 + 24) = v42;
          if ( v13 == -4096 || v17 || v13 == -8192 )
          {
            *(_QWORD *)(v9 + 32) = v43;
            goto LABEL_15;
          }
          sub_BD6050(v9 + 8, v40 & 0xFFFFFFFFFFFFFFF8LL);
          v14 = v42;
          v15 = v43;
          v16 = v42 != -4096;
          v17 = v42 == 0;
        }
        *(_QWORD *)(v9 + 32) = v15;
        v39 = &unk_49DB368;
        if ( v14 != -8192 && !v17 && v16 )
          sub_BD60C0(&v40);
      }
LABEL_15:
      --*(_DWORD *)(v5 + 16);
      ++*(_DWORD *)(v5 + 20);
      v18 = v30;
      v32[0] = 2;
      v33 = a2;
      v32[1] = 0;
      if ( a2 == 0 || a2 == -4096 || a2 == -8192 )
      {
        v34 = v30;
        v31 = &unk_49D9A58;
        v19 = v30;
        v40 = 2;
        v41 = 0;
        v42 = a2;
      }
      else
      {
        sub_BD73F0(v32);
        v31 = &unk_49D9A58;
        v34 = v18;
        v41 = 0;
        v40 = v32[0] & 6;
        v42 = v33;
        if ( v33 == 0 || v33 == -4096 || v33 == -8192 )
        {
          v19 = v18;
        }
        else
        {
          sub_BD6050(&v40, v32[0] & 0xFFFFFFFFFFFFFFF8LL);
          v19 = v34;
        }
      }
      v43 = v19;
      v39 = &unk_49D9A58;
      v44 = v11;
      if ( (unsigned __int8)sub_A570E0(v18, (__int64)&v39, &v25) )
      {
LABEL_21:
        v39 = &unk_49DB368;
        if ( v42 != -4096 && v42 != 0 && v42 != -8192 )
          sub_BD60C0(&v40);
        v31 = &unk_49DB368;
        if ( v33 != 0 && v33 != -4096 && v33 != -8192 )
          sub_BD60C0(v32);
        result = v29;
        goto LABEL_30;
      }
      v21 = *(_DWORD *)(v18 + 24);
      v26 = v25;
      v22 = *(_DWORD *)(v18 + 16);
      ++*(_QWORD *)v18;
      v23 = v22 + 1;
      if ( 4 * v23 >= 3 * v21 )
      {
        v21 *= 2;
      }
      else if ( v21 - *(_DWORD *)(v18 + 20) - v23 > v21 >> 3 )
      {
        goto LABEL_39;
      }
      sub_A631A0(v18, v21);
      sub_A570E0(v18, (__int64)&v39, &v26);
      v23 = *(_DWORD *)(v18 + 16) + 1;
LABEL_39:
      v24 = v26;
      *(_DWORD *)(v18 + 16) = v23;
      v37 = -4096;
      v38 = 0;
      v17 = *(_QWORD *)(v24 + 24) == -4096;
      v36[0] = 2;
      v36[1] = 0;
      if ( !v17 )
      {
        --*(_DWORD *)(v18 + 20);
        v35 = &unk_49DB368;
        if ( v37 != 0 && v37 != -4096 && v37 != -8192 )
          sub_BD60C0(v36);
      }
      sub_A4F610(v24 + 8, &v40);
      *(_QWORD *)(v24 + 32) = v43;
      *(_DWORD *)(v24 + 40) = v44;
      goto LABEL_21;
    }
    v20 = 1;
    while ( v10 != -4096 )
    {
      v8 = (v6 - 1) & (v20 + v8);
      v9 = v7 + 48LL * v8;
      v10 = *(_QWORD *)(v9 + 24);
      if ( v10 == result )
        goto LABEL_6;
      ++v20;
    }
  }
LABEL_30:
  v27 = &unk_49DB368;
  if ( result != 0 && result != -4096 && result != -8192 )
    return sub_BD60C0(v28);
  return result;
}
