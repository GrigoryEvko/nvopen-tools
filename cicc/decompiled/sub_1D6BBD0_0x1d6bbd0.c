// Function: sub_1D6BBD0
// Address: 0x1d6bbd0
//
void __fastcall sub_1D6BBD0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 *v4; // rbx
  __int64 v6; // r14
  __int64 v7; // rax
  char *v8; // r12
  __int64 *v9; // rcx
  __int64 v10; // rdx
  bool v11; // zf
  bool v12; // sf
  bool v13; // of
  __int64 v14; // rsi
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned int v18; // esi
  unsigned int v19; // r11d
  __int64 v20; // rdi
  unsigned int v21; // r8d
  __int64 *v22; // rdx
  __int64 v23; // r10
  int v24; // r8d
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // r10
  int v28; // eax
  __int64 *v29; // r9
  int v30; // edi
  int v31; // edi
  __int64 *v32; // r9
  int v33; // eax
  int v34; // eax
  __int64 *v35; // [rsp+0h] [rbp-70h]
  __int64 *v36; // [rsp+0h] [rbp-70h]
  __int64 *v37; // [rsp+0h] [rbp-70h]
  __int64 *v38; // [rsp+0h] [rbp-70h]
  __int64 v39; // [rsp+8h] [rbp-68h]
  __int64 *v40; // [rsp+10h] [rbp-60h]
  int v41; // [rsp+10h] [rbp-60h]
  int v42; // [rsp+10h] [rbp-60h]
  int v43; // [rsp+10h] [rbp-60h]
  int v44; // [rsp+10h] [rbp-60h]
  __int64 *v45; // [rsp+10h] [rbp-60h]
  __int64 v47; // [rsp+28h] [rbp-48h] BYREF
  __int64 v48; // [rsp+30h] [rbp-40h] BYREF
  _QWORD v49[7]; // [rsp+38h] [rbp-38h] BYREF

  if ( a1 != a2 && a2 != a1 + 2 )
  {
    v4 = a1 + 4;
    v39 = a3 + 832;
    while ( 1 )
    {
      v6 = *a1;
      v7 = *(v4 - 2);
      v8 = (char *)(v4 - 2);
      v9 = v4;
      v10 = a1[1];
      if ( *a1 == v7 )
        goto LABEL_12;
      v13 = __OFSUB__(v10, *(v4 - 1));
      v11 = v10 == *(v4 - 1);
      v12 = v10 - *(v4 - 1) < 0;
      if ( v10 == *(v4 - 1) )
        break;
LABEL_6:
      if ( !(v12 ^ v13 | v11) )
      {
        v14 = *(v4 - 2);
        v15 = *(v4 - 1);
        v16 = (v8 - (char *)a1) >> 4;
        if ( v8 - (char *)a1 > 0 )
        {
          do
          {
            v17 = *((_QWORD *)v8 - 2);
            v8 -= 16;
            *((_QWORD *)v8 + 2) = v17;
            *((_QWORD *)v8 + 3) = *((_QWORD *)v8 + 1);
            --v16;
          }
          while ( v16 );
        }
        *a1 = v14;
        a1[1] = v15;
        goto LABEL_10;
      }
LABEL_12:
      v40 = v9;
      sub_1D6B800(v4 - 2, a3);
      v9 = v40;
LABEL_10:
      v4 += 2;
      if ( a2 == v9 )
        return;
    }
    v18 = *(_DWORD *)(a3 + 856);
    v48 = *(v4 - 2);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a3 + 840);
      v21 = (v18 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v22 = (__int64 *)(v20 + 16LL * v21);
      v23 = *v22;
      if ( v7 == *v22 )
      {
        v24 = *((_DWORD *)v22 + 2);
LABEL_16:
        v47 = v6;
        goto LABEL_17;
      }
      v41 = 1;
      v29 = 0;
      while ( v23 != -8 )
      {
        if ( v29 || v23 != -16 )
          v22 = v29;
        v21 = v19 & (v41 + v21);
        v37 = (__int64 *)(v20 + 16LL * v21);
        v23 = *v37;
        if ( v7 == *v37 )
        {
          v24 = *((_DWORD *)v37 + 2);
          goto LABEL_16;
        }
        ++v41;
        v29 = v22;
        v22 = (__int64 *)(v20 + 16LL * v21);
      }
      v30 = *(_DWORD *)(a3 + 848);
      if ( !v29 )
        v29 = v22;
      ++*(_QWORD *)(a3 + 832);
      v31 = v30 + 1;
      if ( 4 * v31 < 3 * v18 )
      {
        if ( v18 - *(_DWORD *)(a3 + 852) - v31 > v18 >> 3 )
          goto LABEL_26;
        v45 = v4;
        goto LABEL_44;
      }
    }
    else
    {
      ++*(_QWORD *)(a3 + 832);
    }
    v45 = v4;
    v18 *= 2;
LABEL_44:
    sub_1D6B640(v39, v18);
    sub_1D66AA0(v39, &v48, v49);
    v29 = (__int64 *)v49[0];
    v7 = v48;
    v9 = v45;
    v31 = *(_DWORD *)(a3 + 848) + 1;
LABEL_26:
    *(_DWORD *)(a3 + 848) = v31;
    if ( *v29 != -8 )
      --*(_DWORD *)(a3 + 852);
    *v29 = v7;
    *((_DWORD *)v29 + 2) = 0;
    v18 = *(_DWORD *)(a3 + 856);
    v47 = v6;
    v20 = *(_QWORD *)(a3 + 840);
    if ( !v18 )
    {
      ++*(_QWORD *)(a3 + 832);
      v24 = 0;
      goto LABEL_30;
    }
    v24 = 0;
    v19 = v18 - 1;
LABEL_17:
    v25 = v19 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v26 = (__int64 *)(v20 + 16LL * v25);
    v27 = *v26;
    if ( v6 == *v26 )
    {
      v28 = *((_DWORD *)v26 + 2);
LABEL_19:
      v13 = __OFSUB__(v28, v24);
      v11 = v28 == v24;
      v12 = v28 - v24 < 0;
      goto LABEL_6;
    }
    v43 = 1;
    v32 = 0;
    while ( v27 != -8 )
    {
      if ( v32 || v27 != -16 )
        v26 = v32;
      v25 = v19 & (v43 + v25);
      v38 = (__int64 *)(v20 + 16LL * v25);
      v27 = *v38;
      if ( v6 == *v38 )
      {
        v28 = *((_DWORD *)v38 + 2);
        goto LABEL_19;
      }
      ++v43;
      v32 = v26;
      v26 = (__int64 *)(v20 + 16LL * v25);
    }
    if ( !v32 )
      v32 = v26;
    v34 = *(_DWORD *)(a3 + 848);
    ++*(_QWORD *)(a3 + 832);
    v33 = v34 + 1;
    if ( 4 * v33 < 3 * v18 )
    {
      if ( v18 - (v33 + *(_DWORD *)(a3 + 852)) <= v18 >> 3 )
      {
        v36 = v9;
        v44 = v24;
        sub_1D6B640(v39, v18);
        sub_1D66AA0(v39, &v47, v49);
        v32 = (__int64 *)v49[0];
        v6 = v47;
        v9 = v36;
        v24 = v44;
        v33 = *(_DWORD *)(a3 + 848) + 1;
      }
      goto LABEL_31;
    }
LABEL_30:
    v35 = v9;
    v42 = v24;
    sub_1D6B640(v39, 2 * v18);
    sub_1D66AA0(v39, &v47, v49);
    v32 = (__int64 *)v49[0];
    v6 = v47;
    v24 = v42;
    v9 = v35;
    v33 = *(_DWORD *)(a3 + 848) + 1;
LABEL_31:
    *(_DWORD *)(a3 + 848) = v33;
    if ( *v32 != -8 )
      --*(_DWORD *)(a3 + 852);
    *v32 = v6;
    v28 = 0;
    *((_DWORD *)v32 + 2) = 0;
    goto LABEL_19;
  }
}
