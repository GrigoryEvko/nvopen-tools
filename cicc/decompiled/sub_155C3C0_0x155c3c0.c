// Function: sub_155C3C0
// Address: 0x155c3c0
//
__int64 __fastcall sub_155C3C0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // r15
  char v5; // al
  __int64 v6; // rbx
  __int64 v7; // r15
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // rsi
  bool v11; // zf
  bool v12; // r8
  bool v13; // al
  __int64 v14; // rsi
  bool v15; // al
  __int64 v16; // r15
  __int64 v17; // rax
  char v18; // al
  _QWORD *v19; // r12
  __int64 v20; // rax
  __int64 result; // rax
  unsigned int v22; // esi
  int v23; // eax
  int v24; // eax
  __int64 v25; // rdx
  int v26; // [rsp+Ch] [rbp-F4h]
  void *v27; // [rsp+10h] [rbp-F0h] BYREF
  _QWORD v28[2]; // [rsp+18h] [rbp-E8h] BYREF
  __int64 v29; // [rsp+28h] [rbp-D8h]
  __int64 v30; // [rsp+30h] [rbp-D0h]
  void *v31; // [rsp+40h] [rbp-C0h]
  _QWORD v32[2]; // [rsp+48h] [rbp-B8h] BYREF
  __int64 v33; // [rsp+58h] [rbp-A8h]
  __int64 v34; // [rsp+60h] [rbp-A0h]
  _QWORD *v35; // [rsp+70h] [rbp-90h] BYREF
  _QWORD v36[2]; // [rsp+78h] [rbp-88h] BYREF
  __int64 v37; // [rsp+88h] [rbp-78h]
  __int64 v38; // [rsp+90h] [rbp-70h]
  void *v39; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v40; // [rsp+A8h] [rbp-58h] BYREF
  __int64 v41; // [rsp+B0h] [rbp-50h]
  __int64 v42; // [rsp+B8h] [rbp-48h]
  __int64 v43; // [rsp+C0h] [rbp-40h]
  int v44; // [rsp+C8h] [rbp-38h]

  v3 = a1[1];
  v28[1] = 0;
  v28[0] = v3 & 6;
  v29 = a1[3];
  if ( v29 != 0 && v29 != -8 && v29 != -16 )
    sub_1649AC0(v28, v3 & 0xFFFFFFFFFFFFFFF8LL);
  v4 = a1[4];
  v30 = v4;
  v27 = &unk_49ECE00;
  v5 = sub_154CE90(v4, (__int64)&v27, &v39);
  v6 = (__int64)v39;
  if ( !v5 )
    v6 = *(_QWORD *)(v4 + 8) + 48LL * *(unsigned int *)(v4 + 24);
  v7 = v30;
  if ( v6 != *(_QWORD *)(v30 + 8) + 48LL * *(unsigned int *)(v30 + 24) )
  {
    v8 = *(_DWORD *)(v6 + 40);
    v40 = 2;
    v41 = 0;
    v26 = v8;
    v42 = -16;
    v39 = &unk_49ECE00;
    v43 = 0;
    v9 = *(_QWORD *)(v6 + 24);
    if ( v9 == -16 )
    {
      *(_QWORD *)(v6 + 32) = 0;
      goto LABEL_16;
    }
    if ( v9 == -8 || !v9 )
    {
      *(_QWORD *)(v6 + 24) = -16;
    }
    else
    {
      sub_1649B30(v6 + 8);
      v10 = v42;
      v11 = v42 == -8;
      *(_QWORD *)(v6 + 24) = v42;
      v12 = v10 != 0;
      v13 = v10 != -16;
      if ( v10 == 0 || v11 || v10 == -16 )
      {
        v14 = v43;
        v15 = v12 && !v11 && v13;
LABEL_14:
        *(_QWORD *)(v6 + 32) = v14;
        v39 = &unk_49EE2B0;
        if ( v15 )
          sub_1649B30(&v40);
LABEL_16:
        --*(_DWORD *)(v7 + 16);
        ++*(_DWORD *)(v7 + 20);
        v16 = v30;
        v32[0] = 2;
        v33 = a2;
        v32[1] = 0;
        if ( a2 == 0 || a2 == -8 || a2 == -16 )
        {
          v34 = v30;
          v31 = &unk_49ECE00;
          v17 = v30;
          v40 = 2;
          v41 = 0;
          v42 = a2;
        }
        else
        {
          sub_164C220(v32);
          v31 = &unk_49ECE00;
          v34 = v16;
          v41 = 0;
          v40 = v32[0] & 6;
          v42 = v33;
          if ( v33 == 0 || v33 == -8 || v33 == -16 )
          {
            v17 = v16;
          }
          else
          {
            sub_1649AC0(&v40, v32[0] & 0xFFFFFFFFFFFFFFF8LL);
            v17 = v34;
          }
        }
        v43 = v17;
        v39 = &unk_49ECE00;
        v44 = v26;
        v18 = sub_154CE90(v16, (__int64)&v39, &v35);
        v19 = v35;
        if ( v18 )
        {
          v20 = v42;
LABEL_23:
          v39 = &unk_49EE2B0;
          if ( v20 != -8 && v20 != 0 && v20 != -16 )
            sub_1649B30(&v40);
          v31 = &unk_49EE2B0;
          if ( v33 != 0 && v33 != -8 && v33 != -16 )
            sub_1649B30(v32);
          goto LABEL_29;
        }
        v22 = *(_DWORD *)(v16 + 24);
        v23 = *(_DWORD *)(v16 + 16);
        ++*(_QWORD *)v16;
        v24 = v23 + 1;
        if ( 4 * v24 >= 3 * v22 )
        {
          v22 *= 2;
        }
        else if ( v22 - *(_DWORD *)(v16 + 20) - v24 > v22 >> 3 )
        {
          goto LABEL_38;
        }
        sub_1556EF0(v16, v22);
        sub_154CE90(v16, (__int64)&v39, &v35);
        v19 = v35;
        v24 = *(_DWORD *)(v16 + 16) + 1;
LABEL_38:
        *(_DWORD *)(v16 + 16) = v24;
        v36[0] = 2;
        v36[1] = 0;
        v37 = -8;
        v38 = 0;
        if ( v19[3] != -8 )
        {
          --*(_DWORD *)(v16 + 20);
          v35 = &unk_49EE2B0;
          if ( v37 != -8 && v37 != 0 && v37 != -16 )
            sub_1649B30(v36);
        }
        v25 = v19[3];
        v20 = v42;
        if ( v25 != v42 )
        {
          if ( v25 != -8 && v25 != 0 && v25 != -16 )
          {
            sub_1649B30(v19 + 1);
            v20 = v42;
          }
          v19[3] = v20;
          if ( v20 != 0 && v20 != -8 && v20 != -16 )
            sub_1649AC0(v19 + 1, v40 & 0xFFFFFFFFFFFFFFF8LL);
          v20 = v42;
        }
        v19[4] = v43;
        *((_DWORD *)v19 + 10) = v44;
        goto LABEL_23;
      }
      sub_1649AC0(v6 + 8, v40 & 0xFFFFFFFFFFFFFFF8LL);
    }
    v14 = v43;
    v15 = v42 != 0 && v42 != -8 && v42 != -16;
    goto LABEL_14;
  }
LABEL_29:
  v27 = &unk_49EE2B0;
  result = v29;
  if ( v29 != 0 && v29 != -8 && v29 != -16 )
    return sub_1649B30(v28);
  return result;
}
