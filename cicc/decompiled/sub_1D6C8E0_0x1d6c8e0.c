// Function: sub_1D6C8E0
// Address: 0x1d6c8e0
//
char __fastcall sub_1D6C8E0(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdx
  bool v6; // sf
  bool v7; // of
  __int64 v9; // r12
  unsigned int v11; // esi
  __int64 v12; // r14
  unsigned int v13; // edi
  __int64 v14; // rcx
  unsigned int v15; // r8d
  __int64 *v16; // rdx
  __int64 v17; // r10
  int v18; // r13d
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // r9
  int v22; // eax
  __int64 *v23; // r9
  int v24; // ecx
  __int64 *v25; // r8
  int v26; // edx
  int v27; // r10d
  int v28; // eax
  int v29; // r11d
  int v30; // ecx
  __int64 *v31; // r11
  __int64 *v32; // r15
  __int64 v33; // [rsp+8h] [rbp-48h] BYREF
  __int64 v34; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v35[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *a3;
  v4 = *a2;
  v5 = a3[1];
  if ( *a2 != v3 )
  {
    v7 = __OFSUB__(a2[1], v5);
    v6 = a2[1] - v5 < 0;
    if ( a2[1] != v5 )
      return v6 ^ v7;
    v9 = *a1;
    v34 = *a2;
    v11 = *(_DWORD *)(v9 + 856);
    v12 = v9 + 832;
    if ( v11 )
    {
      v13 = v11 - 1;
      v14 = *(_QWORD *)(v9 + 840);
      v15 = (v11 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v16 = (__int64 *)(v14 + 16LL * v15);
      v17 = *v16;
      if ( v4 == *v16 )
      {
        v18 = *((_DWORD *)v16 + 2);
LABEL_7:
        v33 = v3;
        goto LABEL_8;
      }
      v29 = 1;
      v23 = 0;
      while ( v17 != -8 )
      {
        if ( v17 != -16 || v23 )
          v16 = v23;
        v15 = v13 & (v29 + v15);
        v32 = (__int64 *)(v14 + 16LL * v15);
        v17 = *v32;
        if ( v4 == *v32 )
        {
          v18 = *((_DWORD *)v32 + 2);
          goto LABEL_7;
        }
        ++v29;
        v23 = v16;
        v16 = (__int64 *)(v14 + 16LL * v15);
      }
      v30 = *(_DWORD *)(v9 + 848);
      if ( !v23 )
        v23 = v16;
      ++*(_QWORD *)(v9 + 832);
      v24 = v30 + 1;
      if ( 4 * v24 < 3 * v11 )
      {
        if ( v11 - *(_DWORD *)(v9 + 852) - v24 > v11 >> 3 )
          goto LABEL_15;
LABEL_14:
        sub_1D6B640(v9 + 832, v11);
        sub_1D66AA0(v9 + 832, &v34, v35);
        v23 = (__int64 *)v35[0];
        v4 = v34;
        v24 = *(_DWORD *)(v9 + 848) + 1;
LABEL_15:
        *(_DWORD *)(v9 + 848) = v24;
        if ( *v23 != -8 )
          --*(_DWORD *)(v9 + 852);
        *v23 = v4;
        *((_DWORD *)v23 + 2) = 0;
        v9 = *a1;
        v33 = v3;
        v11 = *(_DWORD *)(v9 + 856);
        v14 = *(_QWORD *)(v9 + 840);
        v12 = v9 + 832;
        if ( !v11 )
        {
          ++*(_QWORD *)(v9 + 832);
          v18 = 0;
          goto LABEL_19;
        }
        v18 = 0;
        v13 = v11 - 1;
LABEL_8:
        v19 = v13 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
        v20 = (__int64 *)(v14 + 16LL * v19);
        v21 = *v20;
        if ( v3 == *v20 )
        {
          v22 = *((_DWORD *)v20 + 2);
LABEL_10:
          v7 = __OFSUB__(v18, v22);
          v6 = v18 - v22 < 0;
          return v6 ^ v7;
        }
        v27 = 1;
        v25 = 0;
        while ( v21 != -8 )
        {
          if ( v21 != -16 || v25 )
            v20 = v25;
          v19 = v13 & (v27 + v19);
          v31 = (__int64 *)(v14 + 16LL * v19);
          v21 = *v31;
          if ( v3 == *v31 )
          {
            v22 = *((_DWORD *)v31 + 2);
            goto LABEL_10;
          }
          ++v27;
          v25 = v20;
          v20 = (__int64 *)(v14 + 16LL * v19);
        }
        if ( !v25 )
          v25 = v20;
        v28 = *(_DWORD *)(v9 + 848);
        ++*(_QWORD *)(v9 + 832);
        v26 = v28 + 1;
        if ( 4 * (v28 + 1) < 3 * v11 )
        {
          if ( v11 - *(_DWORD *)(v9 + 852) - v26 > v11 >> 3 )
            goto LABEL_21;
LABEL_20:
          sub_1D6B640(v12, v11);
          sub_1D66AA0(v12, &v33, v35);
          v25 = (__int64 *)v35[0];
          v3 = v33;
          v26 = *(_DWORD *)(v9 + 848) + 1;
LABEL_21:
          *(_DWORD *)(v9 + 848) = v26;
          if ( *v25 != -8 )
            --*(_DWORD *)(v9 + 852);
          *v25 = v3;
          v22 = 0;
          *((_DWORD *)v25 + 2) = 0;
          goto LABEL_10;
        }
LABEL_19:
        v11 *= 2;
        goto LABEL_20;
      }
    }
    else
    {
      ++*(_QWORD *)(v9 + 832);
    }
    v11 *= 2;
    goto LABEL_14;
  }
  return 0;
}
