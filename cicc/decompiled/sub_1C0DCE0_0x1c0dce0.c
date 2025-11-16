// Function: sub_1C0DCE0
// Address: 0x1c0dce0
//
__int64 __fastcall sub_1C0DCE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  int v7; // eax
  __int64 v8; // rdi
  int v9; // eax
  int v10; // r8d
  unsigned int v11; // edx
  __int64 v12; // rcx
  __int64 v13; // rsi
  int v14; // esi
  __int64 v15; // rcx
  unsigned int v16; // edx
  __int64 *v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rsi
  int v20; // r9d
  __int64 v21; // rdi
  unsigned int v22; // ecx
  __int64 *v23; // rdx
  __int64 v24; // r8
  __int64 v25; // rsi
  __int64 v27; // r10
  unsigned int v28; // r11d
  __int64 *v29; // rdx
  __int64 *v30; // r10
  int v31; // r11d
  int v32; // ecx
  int v33; // ecx
  int v34; // r11d
  __int64 *v35; // r10
  int v36; // edx
  int v37; // r10d
  __int64 *v38; // r11
  int v39; // [rsp+Ch] [rbp-84h]
  int v40; // [rsp+18h] [rbp-78h]
  __int64 *v41; // [rsp+18h] [rbp-78h]
  __int64 *v42; // [rsp+18h] [rbp-78h]
  __int64 v43; // [rsp+28h] [rbp-68h] BYREF
  __int64 v44; // [rsp+30h] [rbp-60h] BYREF
  __int64 *v45; // [rsp+38h] [rbp-58h] BYREF
  __int64 v46; // [rsp+40h] [rbp-50h] BYREF
  __int64 v47; // [rsp+48h] [rbp-48h]
  __int64 v48; // [rsp+50h] [rbp-40h]
  __int64 v49; // [rsp+58h] [rbp-38h]

  v3 = a2 + 72;
  v4 = *(_QWORD *)(a2 + 80);
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  if ( v4 != a2 + 72 )
  {
    v7 = 0;
    v8 = 0;
    while ( 1 )
    {
      v13 = v4 - 24;
      if ( !v4 )
        v13 = 0;
      v43 = v13;
      if ( v7 )
      {
        v9 = v7 - 1;
        v10 = 1;
        v11 = v9 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v12 = *(_QWORD *)(v8 + 8LL * v11);
        if ( v13 == v12 )
        {
LABEL_4:
          v4 = *(_QWORD *)(v4 + 8);
          if ( v3 == v4 )
            return j___libc_free_0(v8);
          goto LABEL_5;
        }
        while ( v12 != -8 )
        {
          v11 = v9 & (v10 + v11);
          v12 = *(_QWORD *)(v8 + 8LL * v11);
          if ( v13 == v12 )
            goto LABEL_4;
          ++v10;
        }
      }
      sub_1C0D450(a1, v13, (__int64)&v46);
      v14 = v49;
      if ( !(_DWORD)v49 )
        break;
      v15 = v43;
      v16 = (v49 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
      v17 = (__int64 *)(v47 + 8LL * v16);
      v18 = *v17;
      if ( v43 == *v17 )
        goto LABEL_11;
      v34 = 1;
      v35 = 0;
      while ( v18 != -8 )
      {
        if ( v35 || v18 != -16 )
          v17 = v35;
        v16 = (v49 - 1) & (v34 + v16);
        v42 = (__int64 *)(v47 + 8LL * v16);
        v18 = *v42;
        if ( v43 == *v42 )
          goto LABEL_11;
        ++v34;
        v35 = v17;
        v17 = (__int64 *)(v47 + 8LL * v16);
      }
      if ( !v35 )
        v35 = v17;
      ++v46;
      v36 = v48 + 1;
      if ( 4 * ((int)v48 + 1) >= (unsigned int)(3 * v49) )
        goto LABEL_45;
      if ( (int)v49 - HIDWORD(v48) - v36 <= (unsigned int)v49 >> 3 )
        goto LABEL_46;
LABEL_41:
      LODWORD(v48) = v36;
      if ( *v35 != -8 )
        --HIDWORD(v48);
      *v35 = v15;
      v18 = v43;
LABEL_11:
      v19 = *(unsigned int *)(a1 + 64);
      v44 = v18;
      if ( !(_DWORD)v19 )
        goto LABEL_34;
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 48);
      v22 = (v19 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v23 = (__int64 *)(v21 + 16LL * v22);
      v24 = *v23;
      if ( v18 != *v23 )
      {
        v40 = 1;
        v27 = *v23;
        v28 = (v19 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        while ( 1 )
        {
          if ( v27 == -8 )
            goto LABEL_34;
          v28 = v20 & (v40 + v28);
          v39 = v40 + 1;
          v41 = (__int64 *)(v21 + 16LL * v28);
          v27 = *v41;
          if ( v18 == *v41 )
            break;
          v40 = v39;
        }
        v29 = (__int64 *)(v21 + 16LL * (v20 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4))));
        v30 = 0;
        if ( v41 == (__int64 *)(v21 + 16LL * (unsigned int)v19) )
          goto LABEL_34;
        v31 = 1;
        while ( v24 != -8 )
        {
          if ( v24 != -16 || v30 )
            v29 = v30;
          v37 = v31 + 1;
          v22 = v20 & (v31 + v22);
          v38 = (__int64 *)(v21 + 16LL * v22);
          v24 = *v38;
          if ( v18 == *v38 )
          {
            v25 = v38[1];
            goto LABEL_15;
          }
          v31 = v37;
          v30 = v29;
          v29 = (__int64 *)(v21 + 16LL * v22);
        }
        v32 = *(_DWORD *)(a1 + 56);
        if ( !v30 )
          v30 = v29;
        ++*(_QWORD *)(a1 + 40);
        v33 = v32 + 1;
        if ( 4 * v33 >= (unsigned int)(3 * v19) )
        {
          LODWORD(v19) = 2 * v19;
        }
        else if ( (int)v19 - *(_DWORD *)(a1 + 60) - v33 > (unsigned int)v19 >> 3 )
        {
LABEL_31:
          *(_DWORD *)(a1 + 56) = v33;
          if ( *v30 != -8 )
            --*(_DWORD *)(a1 + 60);
          *v30 = v18;
          v30[1] = 0;
          goto LABEL_34;
        }
        sub_1C04E30(a1 + 40, v19);
        sub_1C09800(a1 + 40, &v44, &v45);
        v30 = v45;
        v18 = v44;
        v33 = *(_DWORD *)(a1 + 56) + 1;
        goto LABEL_31;
      }
      if ( v23 == (__int64 *)(v21 + 16 * v19) )
LABEL_34:
        v25 = 0;
      else
        v25 = v23[1];
LABEL_15:
      sub_1C0BEB0(a1, v25, (__int64)&v46, a3);
      v4 = *(_QWORD *)(v4 + 8);
      v8 = v47;
      if ( v3 == v4 )
        return j___libc_free_0(v8);
LABEL_5:
      v7 = v49;
    }
    ++v46;
LABEL_45:
    v14 = 2 * v49;
LABEL_46:
    sub_13B3D40((__int64)&v46, v14);
    sub_1898220((__int64)&v46, &v43, &v45);
    v35 = v45;
    v15 = v43;
    v36 = v48 + 1;
    goto LABEL_41;
  }
  v8 = 0;
  return j___libc_free_0(v8);
}
