// Function: sub_1D6B800
// Address: 0x1d6b800
//
void __fastcall sub_1D6B800(__int64 *a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v3; // r14
  __int64 v4; // r12
  __int64 v5; // r15
  __int64 v6; // rdx
  __int64 *v8; // r11
  unsigned int v9; // esi
  unsigned int v10; // r8d
  __int64 v11; // rdi
  unsigned int v12; // r9d
  __int64 *v13; // rax
  __int64 v14; // rcx
  int v15; // ecx
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // r10
  int v19; // eax
  bool v20; // sf
  bool v21; // of
  __int64 *v22; // r9
  int v23; // eax
  int v24; // eax
  int v25; // r10d
  __int64 *v26; // rdx
  int v27; // eax
  int v28; // ecx
  __int64 v29; // rdi
  int v30; // eax
  int v31; // edx
  __int64 *v32; // r10
  __int64 *v33; // [rsp+0h] [rbp-70h]
  __int64 *v34; // [rsp+0h] [rbp-70h]
  __int64 *v35; // [rsp+0h] [rbp-70h]
  int v36; // [rsp+8h] [rbp-68h]
  int v37; // [rsp+8h] [rbp-68h]
  __int64 *v38; // [rsp+8h] [rbp-68h]
  int v39; // [rsp+8h] [rbp-68h]
  __int64 v40; // [rsp+10h] [rbp-60h]
  __int64 v41; // [rsp+28h] [rbp-48h] BYREF
  __int64 v42; // [rsp+30h] [rbp-40h] BYREF
  _QWORD v43[7]; // [rsp+38h] [rbp-38h] BYREF

  v2 = a1;
  v3 = *a1;
  v4 = *(a1 - 2);
  v5 = a1[1];
  v6 = *(a1 - 1);
  if ( *a1 != v4 )
  {
    v2 = a1 - 2;
    v40 = a2 + 832;
    while ( 1 )
    {
      v8 = v2 + 2;
      if ( v5 != v6 )
      {
        if ( v5 >= v6 )
          goto LABEL_14;
        goto LABEL_4;
      }
      v9 = *(_DWORD *)(a2 + 856);
      v42 = v3;
      if ( v9 )
      {
        v10 = v9 - 1;
        v11 = *(_QWORD *)(a2 + 840);
        v12 = (v9 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
        v13 = (__int64 *)(v11 + 16LL * v12);
        v14 = *v13;
        if ( v3 == *v13 )
        {
          v15 = *((_DWORD *)v13 + 2);
LABEL_10:
          v41 = v4;
          goto LABEL_11;
        }
        v25 = 1;
        v26 = 0;
        while ( v14 != -8 )
        {
          if ( v14 != -16 || v26 )
            v13 = v26;
          v31 = v25 + 1;
          v12 = v10 & (v25 + v12);
          v32 = (__int64 *)(v11 + 16LL * v12);
          v14 = *v32;
          if ( v3 == *v32 )
          {
            v15 = *((_DWORD *)v32 + 2);
            goto LABEL_10;
          }
          v25 = v31;
          v26 = v13;
          v13 = (__int64 *)(v11 + 16LL * v12);
        }
        if ( !v26 )
          v26 = v13;
        v27 = *(_DWORD *)(a2 + 848);
        ++*(_QWORD *)(a2 + 832);
        v28 = v27 + 1;
        if ( 4 * (v27 + 1) < 3 * v9 )
        {
          v29 = v3;
          if ( v9 - *(_DWORD *)(a2 + 852) - v28 > v9 >> 3 )
            goto LABEL_32;
          v38 = v2 + 2;
          goto LABEL_40;
        }
      }
      else
      {
        ++*(_QWORD *)(a2 + 832);
      }
      v38 = v2 + 2;
      v9 *= 2;
LABEL_40:
      sub_1D6B640(v40, v9);
      sub_1D66AA0(v40, &v42, v43);
      v26 = (__int64 *)v43[0];
      v29 = v42;
      v8 = v38;
      v28 = *(_DWORD *)(a2 + 848) + 1;
LABEL_32:
      *(_DWORD *)(a2 + 848) = v28;
      if ( *v26 != -8 )
        --*(_DWORD *)(a2 + 852);
      *v26 = v29;
      *((_DWORD *)v26 + 2) = 0;
      v9 = *(_DWORD *)(a2 + 856);
      v41 = v4;
      v11 = *(_QWORD *)(a2 + 840);
      if ( !v9 )
      {
        ++*(_QWORD *)(a2 + 832);
        v15 = 0;
        goto LABEL_36;
      }
      v15 = 0;
      v10 = v9 - 1;
LABEL_11:
      v16 = v10 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v17 = (__int64 *)(v11 + 16LL * v16);
      v18 = *v17;
      if ( v4 == *v17 )
      {
        v19 = *((_DWORD *)v17 + 2);
        v21 = __OFSUB__(v15, v19);
        v20 = v15 - v19 < 0;
        goto LABEL_13;
      }
      v36 = 1;
      v22 = 0;
      while ( v18 != -8 )
      {
        if ( v18 != -16 || v22 )
          v17 = v22;
        v16 = v10 & (v36 + v16);
        v35 = (__int64 *)(v11 + 16LL * v16);
        v18 = *v35;
        if ( *v35 == v4 )
        {
          v30 = *((_DWORD *)v35 + 2);
          v21 = __OFSUB__(v15, v30);
          v20 = v15 - v30 < 0;
          goto LABEL_13;
        }
        ++v36;
        v22 = v17;
        v17 = (__int64 *)(v11 + 16LL * v16);
      }
      if ( !v22 )
        v22 = v17;
      v23 = *(_DWORD *)(a2 + 848);
      ++*(_QWORD *)(a2 + 832);
      v24 = v23 + 1;
      if ( 4 * v24 < 3 * v9 )
      {
        if ( v9 - (v24 + *(_DWORD *)(a2 + 852)) <= v9 >> 3 )
        {
          v34 = v8;
          v39 = v15;
          sub_1D6B640(v40, v9);
          sub_1D66AA0(v40, &v41, v43);
          v22 = (__int64 *)v43[0];
          v4 = v41;
          v8 = v34;
          v15 = v39;
          v24 = *(_DWORD *)(a2 + 848) + 1;
        }
        goto LABEL_23;
      }
LABEL_36:
      v33 = v8;
      v37 = v15;
      sub_1D6B640(v40, 2 * v9);
      sub_1D66AA0(v40, &v41, v43);
      v22 = (__int64 *)v43[0];
      v4 = v41;
      v15 = v37;
      v8 = v33;
      v24 = *(_DWORD *)(a2 + 848) + 1;
LABEL_23:
      *(_DWORD *)(a2 + 848) = v24;
      if ( *v22 != -8 )
        --*(_DWORD *)(a2 + 852);
      *v22 = v4;
      *((_DWORD *)v22 + 2) = 0;
      v21 = 0;
      v20 = v15 < 0;
LABEL_13:
      if ( v20 == v21 )
      {
LABEL_14:
        v2 = v8;
        break;
      }
LABEL_4:
      v4 = *(v2 - 2);
      v6 = *(v2 - 1);
      v2[2] = *v2;
      v2[3] = v2[1];
      if ( v3 == v4 )
        break;
      v2 -= 2;
    }
  }
  *v2 = v3;
  v2[1] = v5;
}
