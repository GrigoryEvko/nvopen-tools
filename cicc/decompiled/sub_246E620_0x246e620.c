// Function: sub_246E620
// Address: 0x246e620
//
__int64 __fastcall sub_246E620(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 result; // rax
  __int64 v5; // r14
  __int64 v6; // rdx
  __int64 v7; // rsi
  unsigned int v8; // ecx
  _QWORD *v9; // rbx
  __int64 v10; // rdi
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  bool v16; // al
  bool v17; // zf
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // rax
  int v21; // r9d
  unsigned int v22; // esi
  int v23; // eax
  int v24; // eax
  _QWORD *v25; // r12
  __int64 v26; // rdx
  _QWORD *v27; // [rsp+10h] [rbp-100h] BYREF
  _QWORD *v28; // [rsp+18h] [rbp-F8h] BYREF
  void *v29; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v30[2]; // [rsp+28h] [rbp-E8h] BYREF
  __int64 v31; // [rsp+38h] [rbp-D8h]
  __int64 v32; // [rsp+40h] [rbp-D0h]
  void *v33; // [rsp+50h] [rbp-C0h]
  _QWORD v34[2]; // [rsp+58h] [rbp-B8h] BYREF
  __int64 v35; // [rsp+68h] [rbp-A8h]
  __int64 v36; // [rsp+70h] [rbp-A0h]
  void *v37; // [rsp+80h] [rbp-90h]
  _QWORD v38[2]; // [rsp+88h] [rbp-88h] BYREF
  __int64 v39; // [rsp+98h] [rbp-78h]
  __int64 v40; // [rsp+A0h] [rbp-70h]
  void *v41; // [rsp+B0h] [rbp-60h] BYREF
  unsigned __int64 v42; // [rsp+B8h] [rbp-58h] BYREF
  __int64 v43; // [rsp+C0h] [rbp-50h]
  __int64 v44; // [rsp+C8h] [rbp-48h]
  __int64 v45; // [rsp+D0h] [rbp-40h]
  __int64 v46; // [rsp+D8h] [rbp-38h]

  v3 = a1[1];
  v30[1] = 0;
  v30[0] = v3 & 6;
  v31 = a1[3];
  result = v31;
  if ( v31 != -4096 && v31 != 0 && v31 != -8192 )
  {
    sub_BD6050(v30, v3 & 0xFFFFFFFFFFFFFFF8LL);
    result = v31;
  }
  v5 = a1[4];
  v32 = v5;
  v29 = &unk_4A16A38;
  v6 = *(unsigned int *)(v5 + 24);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD *)(v5 + 8);
    v8 = (v6 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
    v9 = (_QWORD *)(v7 + 48LL * v8);
    v10 = v9[3];
    if ( v10 == result )
    {
LABEL_6:
      if ( v9 == (_QWORD *)(v7 + 48 * v6) )
        goto LABEL_31;
      v11 = v9[5];
      v42 = 2;
      v43 = 0;
      v44 = -8192;
      v41 = &unk_4A16A38;
      v45 = 0;
      v12 = v9[3];
      if ( v12 == -8192 )
      {
        v9[4] = 0;
      }
      else
      {
        if ( !v12 || v12 == -4096 )
        {
          v9[3] = -8192;
          v14 = v44;
          v15 = v45;
          v16 = v44 != -4096;
          v17 = v44 == 0;
        }
        else
        {
          sub_BD60C0(v9 + 1);
          v13 = v44;
          v17 = v44 == 0;
          v9[3] = v44;
          if ( v13 == -4096 || v17 || v13 == -8192 )
          {
            v9[4] = v45;
            goto LABEL_15;
          }
          sub_BD6050(v9 + 1, v42 & 0xFFFFFFFFFFFFFFF8LL);
          v14 = v44;
          v15 = v45;
          v16 = v44 != 0;
          v17 = v44 == -4096;
        }
        v9[4] = v15;
        v41 = &unk_49DB368;
        if ( v14 != -8192 && !v17 && v16 )
          sub_BD60C0(&v42);
      }
LABEL_15:
      --*(_DWORD *)(v5 + 16);
      ++*(_DWORD *)(v5 + 20);
      v18 = v32;
      v34[0] = 2;
      v35 = a2;
      v34[1] = 0;
      if ( a2 == 0 || a2 == -4096 || a2 == -8192 )
      {
        v36 = v32;
        v33 = &unk_4A16A38;
        v19 = v32;
        v42 = 2;
        v43 = 0;
        v44 = a2;
      }
      else
      {
        sub_BD73F0((__int64)v34);
        v33 = &unk_4A16A38;
        v36 = v18;
        v43 = 0;
        v42 = v34[0] & 6;
        v44 = v35;
        if ( v35 == 0 || v35 == -4096 || v35 == -8192 )
        {
          v19 = v18;
        }
        else
        {
          sub_BD6050(&v42, v34[0] & 0xFFFFFFFFFFFFFFF8LL);
          v19 = v36;
        }
      }
      v45 = v19;
      v41 = &unk_4A16A38;
      v46 = v11;
      if ( (unsigned __int8)sub_246CEE0(v18, (__int64)&v41, &v27) )
      {
        v20 = v44;
LABEL_22:
        v41 = &unk_49DB368;
        if ( v20 != 0 && v20 != -4096 && v20 != -8192 )
          sub_BD60C0(&v42);
        v33 = &unk_49DB368;
        if ( v35 != 0 && v35 != -4096 && v35 != -8192 )
          sub_BD60C0(v34);
        result = v31;
        goto LABEL_31;
      }
      v22 = *(_DWORD *)(v18 + 24);
      v28 = v27;
      v23 = *(_DWORD *)(v18 + 16);
      ++*(_QWORD *)v18;
      v24 = v23 + 1;
      if ( 4 * v24 >= 3 * v22 )
      {
        v22 *= 2;
      }
      else if ( v22 - *(_DWORD *)(v18 + 20) - v24 > v22 >> 3 )
      {
        goto LABEL_40;
      }
      sub_246E1D0(v18, v22);
      sub_246CEE0(v18, (__int64)&v41, &v28);
      v24 = *(_DWORD *)(v18 + 16) + 1;
LABEL_40:
      v25 = v28;
      *(_DWORD *)(v18 + 16) = v24;
      v39 = -4096;
      v40 = 0;
      v17 = v25[3] == -4096;
      v38[0] = 2;
      v38[1] = 0;
      if ( !v17 )
      {
        --*(_DWORD *)(v18 + 20);
        v37 = &unk_49DB368;
        if ( v39 != -4096 && v39 != -8192 )
        {
          if ( v39 )
            sub_BD60C0(v38);
        }
      }
      v26 = v25[3];
      v20 = v44;
      if ( v26 != v44 )
      {
        if ( v26 != -4096 && v26 != 0 && v26 != -8192 )
        {
          sub_BD60C0(v25 + 1);
          v20 = v44;
        }
        v25[3] = v20;
        if ( v20 != 0 && v20 != -4096 && v20 != -8192 )
          sub_BD6050(v25 + 1, v42 & 0xFFFFFFFFFFFFFFF8LL);
        v20 = v44;
      }
      v25[4] = v45;
      v25[5] = v46;
      goto LABEL_22;
    }
    v21 = 1;
    while ( v10 != -4096 )
    {
      v8 = (v6 - 1) & (v21 + v8);
      v9 = (_QWORD *)(v7 + 48LL * v8);
      v10 = v9[3];
      if ( v10 == result )
        goto LABEL_6;
      ++v21;
    }
  }
LABEL_31:
  v29 = &unk_49DB368;
  if ( result != 0 && result != -4096 && result != -8192 )
    return sub_BD60C0(v30);
  return result;
}
