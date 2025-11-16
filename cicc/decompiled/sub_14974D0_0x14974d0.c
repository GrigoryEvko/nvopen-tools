// Function: sub_14974D0
// Address: 0x14974d0
//
__int64 __fastcall sub_14974D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rax
  int v7; // eax
  char v8; // al
  __int64 v9; // r15
  __int64 v10; // rcx
  char v11; // r14
  __int64 v12; // rax
  __int64 v13; // r13
  unsigned int v15; // esi
  int v16; // eax
  int v17; // eax
  __int64 v18; // rdx
  __int64 v19; // r13
  __int64 v20; // [rsp+8h] [rbp-A8h]
  __int64 v21; // [rsp+8h] [rbp-A8h]
  __int64 v22; // [rsp+18h] [rbp-98h] BYREF
  void *v23; // [rsp+20h] [rbp-90h]
  _QWORD v24[2]; // [rsp+28h] [rbp-88h] BYREF
  __int64 v25; // [rsp+38h] [rbp-78h]
  __int64 v26; // [rsp+40h] [rbp-70h]
  void *v27; // [rsp+50h] [rbp-60h] BYREF
  __int64 v28; // [rsp+58h] [rbp-58h] BYREF
  __int64 v29; // [rsp+60h] [rbp-50h]
  __int64 v30; // [rsp+68h] [rbp-48h]
  __int64 v31; // [rsp+70h] [rbp-40h]
  int v32; // [rsp+78h] [rbp-38h]

  v5 = *(_QWORD *)a3;
  v24[0] = 2;
  v24[1] = 0;
  v25 = v5;
  if ( v5 == 0 || v5 == -8 || v5 == -16 )
  {
    v26 = a2;
    v28 = 2;
    v29 = 0;
    v23 = &unk_49EC740;
    v30 = v5;
    v6 = a2;
  }
  else
  {
    sub_164C220(v24);
    v26 = a2;
    v29 = 0;
    v23 = &unk_49EC740;
    v30 = v25;
    v28 = v24[0] & 6;
    if ( v25 == -8 || v25 == 0 || v25 == -16 )
    {
      v6 = a2;
    }
    else
    {
      sub_1649AC0(&v28, v24[0] & 0xFFFFFFFFFFFFFFF8LL);
      v6 = v26;
    }
  }
  v31 = v6;
  v7 = *(_DWORD *)(a3 + 8);
  v27 = &unk_49EC740;
  v32 = v7;
  v8 = sub_14663D0(a2, (__int64)&v27, &v22);
  v9 = v22;
  if ( !v8 )
  {
    v15 = *(_DWORD *)(a2 + 24);
    v16 = *(_DWORD *)(a2 + 16);
    ++*(_QWORD *)a2;
    v17 = v16 + 1;
    if ( 4 * v17 >= 3 * v15 )
    {
      v15 *= 2;
    }
    else if ( v15 - *(_DWORD *)(a2 + 20) - v17 > v15 >> 3 )
    {
      goto LABEL_18;
    }
    sub_14970D0(a2, v15);
    sub_14663D0(a2, (__int64)&v27, &v22);
    v9 = v22;
    v17 = *(_DWORD *)(a2 + 16) + 1;
LABEL_18:
    *(_DWORD *)(a2 + 16) = v17;
    if ( *(_QWORD *)(v9 + 24) == -8 )
    {
      v18 = v30;
      v12 = -8;
      v19 = v9 + 8;
      if ( v30 != -8 )
      {
LABEL_23:
        *(_QWORD *)(v9 + 24) = v18;
        if ( v18 != -8 && v18 != 0 && v18 != -16 )
          sub_1649AC0(v19, v28 & 0xFFFFFFFFFFFFFFF8LL);
        v12 = v30;
      }
    }
    else
    {
      --*(_DWORD *)(a2 + 20);
      v18 = v30;
      v12 = *(_QWORD *)(v9 + 24);
      if ( v30 != v12 )
      {
        v19 = v9 + 8;
        if ( v12 != -8 && v12 != 0 && v12 != -16 )
        {
          sub_1649B30(v9 + 8);
          v18 = v30;
        }
        goto LABEL_23;
      }
    }
    v11 = 1;
    *(_QWORD *)(v9 + 32) = v31;
    *(_DWORD *)(v9 + 40) = v32;
    v10 = *(_QWORD *)a2;
    v13 = *(_QWORD *)(a2 + 8) + 48LL * *(unsigned int *)(a2 + 24);
    goto LABEL_9;
  }
  v10 = *(_QWORD *)a2;
  v11 = 0;
  v12 = v30;
  v13 = *(_QWORD *)(a2 + 8) + 48LL * *(unsigned int *)(a2 + 24);
LABEL_9:
  v27 = &unk_49EE2B0;
  if ( v12 != 0 && v12 != -8 && v12 != -16 )
  {
    v20 = v10;
    sub_1649B30(&v28);
    v10 = v20;
  }
  v23 = &unk_49EE2B0;
  if ( v25 != 0 && v25 != -8 && v25 != -16 )
  {
    v21 = v10;
    sub_1649B30(v24);
    v10 = v21;
  }
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v9;
  *(_QWORD *)(a1 + 24) = v13;
  *(_BYTE *)(a1 + 32) = v11;
  *(_QWORD *)(a1 + 8) = v10;
  return a1;
}
