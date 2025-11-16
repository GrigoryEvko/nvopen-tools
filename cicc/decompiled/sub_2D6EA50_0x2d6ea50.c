// Function: sub_2D6EA50
// Address: 0x2d6ea50
//
__int64 __fastcall sub_2D6EA50(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // r14
  __int64 v4; // rsi
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 v7; // rax
  _QWORD *v8; // r12
  __int64 v9; // r14
  bool v10; // r13
  __int64 v11; // r13
  __int64 v12; // rax
  _QWORD *v13; // rbx
  unsigned int v14; // esi
  __int64 v15; // r8
  __int64 v16; // r15
  int v17; // edx
  int v18; // r15d
  __int64 result; // rax
  __int64 v20; // r9
  unsigned int v21; // edx
  __int64 v22; // rax
  __int64 v23; // rdi
  unsigned int v24; // r8d
  int v25; // esi
  __int64 v26; // rsi
  __int64 v27; // r13
  int v28; // edx
  int v29; // eax
  __int64 v30; // r9
  unsigned int v31; // edx
  __int64 v32; // rax
  __int64 v33; // rdi
  int v34; // ecx
  int v35; // eax
  int v36; // ecx
  int v37; // eax
  __int64 v38; // [rsp+0h] [rbp-B0h]
  __int64 v39; // [rsp+18h] [rbp-98h] BYREF
  _QWORD v40[2]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v41; // [rsp+30h] [rbp-80h]
  _QWORD v42[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v43; // [rsp+50h] [rbp-60h]
  unsigned __int64 v44[2]; // [rsp+60h] [rbp-50h] BYREF
  __int64 v45; // [rsp+70h] [rbp-40h]
  __int64 v46; // [rsp+78h] [rbp-38h]

  v3 = a1;
  v4 = a1[2];
  v44[0] = 0;
  v44[1] = 0;
  v45 = v4;
  if ( v4 != -4096 && v4 != 0 && v4 != -8192 )
  {
    sub_BD6050(v44, *a1 & 0xFFFFFFFFFFFFFFF8LL);
    v4 = v45;
  }
  v5 = a1[3];
  v6 = *(a1 - 2);
  v7 = *(a1 - 1);
  v46 = v5;
  if ( v6 != v4 )
  {
    v8 = a1;
    v9 = a2;
    v38 = a2 + 728;
    while ( 1 )
    {
      v13 = v8;
      v8 -= 4;
      if ( v7 != v5 )
      {
        v10 = v7 > v5;
LABEL_7:
        if ( !v10 )
          goto LABEL_53;
        goto LABEL_8;
      }
      v42[0] = 0;
      v42[1] = 0;
      v43 = v4;
      if ( v4 != 0 && v4 != -4096 && v4 != -8192 )
        sub_BD73F0((__int64)v42);
      v14 = *(_DWORD *)(v9 + 752);
      if ( v14 )
      {
        v15 = v43;
        v20 = *(_QWORD *)(v9 + 736);
        v21 = (v14 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
        v22 = v20 + 32LL * v21;
        v23 = *(_QWORD *)(v22 + 16);
        if ( v23 == v43 )
        {
LABEL_34:
          v18 = *(_DWORD *)(v22 + 24);
          goto LABEL_35;
        }
        v34 = 1;
        v16 = 0;
        while ( v23 != -4096 )
        {
          if ( v23 == -8192 && !v16 )
            v16 = v22;
          v21 = (v14 - 1) & (v34 + v21);
          v22 = v20 + 32LL * v21;
          v23 = *(_QWORD *)(v22 + 16);
          if ( v43 == v23 )
            goto LABEL_34;
          ++v34;
        }
        if ( !v16 )
          v16 = v22;
        v35 = *(_DWORD *)(v9 + 744);
        ++*(_QWORD *)(v9 + 728);
        v17 = v35 + 1;
        v40[0] = v16;
        if ( 4 * (v35 + 1) < 3 * v14 )
        {
          if ( v14 - *(_DWORD *)(v9 + 748) - v17 > v14 >> 3 )
            goto LABEL_25;
          goto LABEL_24;
        }
      }
      else
      {
        ++*(_QWORD *)(v9 + 728);
        v40[0] = 0;
      }
      v14 *= 2;
LABEL_24:
      sub_2D6E640(v38, v14);
      sub_2D67BB0(v38, (__int64)v42, v40);
      v15 = v43;
      v16 = v40[0];
      v17 = *(_DWORD *)(v9 + 744) + 1;
LABEL_25:
      *(_DWORD *)(v9 + 744) = v17;
      if ( *(_QWORD *)(v16 + 16) != -4096 )
        --*(_DWORD *)(v9 + 748);
      sub_2D57220((_QWORD *)v16, v15);
      *(_DWORD *)(v16 + 24) = 0;
      v18 = 0;
LABEL_35:
      v41 = v6;
      v40[0] = 0;
      v40[1] = 0;
      if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
        sub_BD73F0((__int64)v40);
      v24 = *(_DWORD *)(v9 + 752);
      if ( !v24 )
      {
        ++*(_QWORD *)(v9 + 728);
        v39 = 0;
LABEL_40:
        v25 = 2 * v24;
        goto LABEL_41;
      }
      v26 = v41;
      v30 = *(_QWORD *)(v9 + 736);
      v31 = (v24 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
      v32 = v30 + 32LL * v31;
      v33 = *(_QWORD *)(v32 + 16);
      if ( v41 == v33 )
      {
LABEL_46:
        v29 = *(_DWORD *)(v32 + 24);
        goto LABEL_47;
      }
      v36 = 1;
      v27 = 0;
      while ( v33 != -4096 )
      {
        if ( !v27 && v33 == -8192 )
          v27 = v32;
        v31 = (v24 - 1) & (v36 + v31);
        v32 = v30 + 32LL * v31;
        v33 = *(_QWORD *)(v32 + 16);
        if ( v41 == v33 )
          goto LABEL_46;
        ++v36;
      }
      if ( !v27 )
        v27 = v32;
      v37 = *(_DWORD *)(v9 + 744);
      ++*(_QWORD *)(v9 + 728);
      v28 = v37 + 1;
      v39 = v27;
      if ( 4 * (v37 + 1) >= 3 * v24 )
        goto LABEL_40;
      if ( v24 - *(_DWORD *)(v9 + 748) - v28 > v24 >> 3 )
        goto LABEL_42;
      v25 = v24;
LABEL_41:
      sub_2D6E640(v38, v25);
      sub_2D67BB0(v38, (__int64)v40, &v39);
      v26 = v41;
      v27 = v39;
      v28 = *(_DWORD *)(v9 + 744) + 1;
LABEL_42:
      *(_DWORD *)(v9 + 744) = v28;
      if ( *(_QWORD *)(v27 + 16) != -4096 )
        --*(_DWORD *)(v9 + 748);
      sub_2D57220((_QWORD *)v27, v26);
      *(_DWORD *)(v27 + 24) = 0;
      v29 = 0;
      v26 = v41;
LABEL_47:
      v10 = v29 > v18;
      if ( v26 != -4096 && v26 != 0 && v26 != -8192 )
        sub_BD60C0(v40);
      if ( v43 == 0 || v43 == -4096 || v43 == -8192 )
        goto LABEL_7;
      sub_BD60C0(v42);
      if ( !v10 )
      {
LABEL_53:
        v4 = v45;
        v3 = v13;
        break;
      }
LABEL_8:
      v11 = *(v13 - 2);
      v12 = v13[2];
      if ( v11 != v12 )
      {
        if ( v12 != -4096 && v12 != 0 && v12 != -8192 )
          sub_BD60C0(v13);
        v13[2] = v11;
        if ( v11 != -4096 && v11 != 0 && v11 != -8192 )
          sub_BD73F0((__int64)v13);
      }
      v6 = *(v13 - 6);
      v4 = v45;
      v13[3] = *(v13 - 1);
      v7 = *(v13 - 5);
      if ( v6 == v4 )
      {
        v3 = v8;
        break;
      }
      v5 = v46;
    }
  }
  sub_2D57220(v3, v4);
  v3[3] = v46;
  result = v45;
  if ( v45 != 0 && v45 != -4096 && v45 != -8192 )
    return sub_BD60C0(v44);
  return result;
}
