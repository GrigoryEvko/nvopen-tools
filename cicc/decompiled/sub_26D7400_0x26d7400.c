// Function: sub_26D7400
// Address: 0x26d7400
//
__int64 *__fastcall sub_26D7400(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 *v6; // r13
  __int64 v7; // r15
  __int64 *v9; // r12
  char v10; // al
  char v11; // r8
  __int64 v12; // rax
  int v13; // edx
  __int64 v14; // r9
  int v15; // edx
  unsigned int v16; // ecx
  __int64 *v17; // rax
  __int64 v18; // r10
  __int64 v19; // r10
  __int64 v20; // rcx
  unsigned int v21; // esi
  __int64 *v22; // rax
  __int64 v23; // r11
  bool v24; // al
  unsigned int v25; // esi
  __int64 *v26; // r15
  __int64 v27; // r9
  int v28; // r11d
  __int64 *v29; // rdi
  unsigned int v30; // edx
  _QWORD *v31; // rax
  __int64 v32; // r8
  __int64 **v33; // rax
  _QWORD *v34; // rax
  _QWORD *v35; // rdx
  unsigned __int64 v36; // rax
  __int64 v37; // rax
  __int64 *result; // rax
  int v39; // eax
  int v40; // eax
  __int64 v41; // rbx
  int v42; // eax
  int v43; // edx
  int v44; // r15d
  int v45; // esi
  __int64 v46; // [rsp+8h] [rbp-98h]
  __int64 v47; // [rsp+18h] [rbp-88h]
  unsigned __int64 v48; // [rsp+20h] [rbp-80h]
  __int64 v49; // [rsp+28h] [rbp-78h] BYREF
  __int64 *v50; // [rsp+30h] [rbp-70h] BYREF
  __int64 v51; // [rsp+38h] [rbp-68h] BYREF
  _QWORD v52[12]; // [rsp+40h] [rbp-60h] BYREF

  v6 = a3;
  v7 = a1 + 40;
  v9 = &a3[a4];
  v49 = a2;
  v46 = a1 + 968;
  v50 = (__int64 *)*sub_26D72C0(a1 + 968, &v49);
  v48 = *sub_26CC460(a1 + 40, (__int64 *)&v50);
  if ( v9 != v6 )
  {
    v47 = a1 + 40;
    while ( 1 )
    {
      v51 = *v6;
      sub_B19AA0(a5, v51, v49);
      v11 = v10;
      v12 = *(_QWORD *)(a1 + 1016);
      v13 = *(_DWORD *)(v12 + 24);
      v14 = *(_QWORD *)(v12 + 8);
      if ( v13 )
      {
        v15 = v13 - 1;
        v16 = v15 & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
        v17 = (__int64 *)(v14 + 16LL * v16);
        v18 = *v17;
        if ( v49 == *v17 )
        {
LABEL_5:
          v19 = v17[1];
        }
        else
        {
          v40 = 1;
          while ( v18 != -4096 )
          {
            v45 = v40 + 1;
            v16 = v15 & (v40 + v16);
            v17 = (__int64 *)(v14 + 16LL * v16);
            v18 = *v17;
            if ( v49 == *v17 )
              goto LABEL_5;
            v40 = v45;
          }
          v19 = 0;
        }
        v20 = v51;
        v21 = v15 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
        v22 = (__int64 *)(v14 + 16LL * v21);
        v23 = *v22;
        if ( v51 == *v22 )
        {
LABEL_7:
          v24 = v22[1] == v19;
        }
        else
        {
          v39 = 1;
          while ( v23 != -4096 )
          {
            v44 = v39 + 1;
            v21 = v15 & (v39 + v21);
            v22 = (__int64 *)(v14 + 16LL * v21);
            v23 = *v22;
            if ( *v22 == v51 )
              goto LABEL_7;
            v39 = v44;
          }
          v24 = v19 == 0;
        }
      }
      else
      {
        v20 = v51;
        v24 = 1;
      }
      if ( v49 == v20 || !v11 || !v24 )
        goto LABEL_23;
      v25 = *(_DWORD *)(a1 + 992);
      v26 = v50;
      if ( !v25 )
        break;
      v27 = *(_QWORD *)(a1 + 976);
      v28 = 1;
      v29 = 0;
      v30 = (v25 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v31 = (_QWORD *)(v27 + 16LL * v30);
      v32 = *v31;
      if ( *v31 == v20 )
      {
LABEL_13:
        v33 = (__int64 **)(v31 + 1);
        goto LABEL_14;
      }
      while ( v32 != -4096 )
      {
        if ( v32 == -8192 && !v29 )
          v29 = v31;
        v30 = (v25 - 1) & (v28 + v30);
        v31 = (_QWORD *)(v27 + 16LL * v30);
        v32 = *v31;
        if ( *v31 == v20 )
          goto LABEL_13;
        ++v28;
      }
      if ( !v29 )
        v29 = v31;
      v42 = *(_DWORD *)(a1 + 984);
      ++*(_QWORD *)(a1 + 968);
      v43 = v42 + 1;
      v52[0] = v29;
      if ( 4 * (v42 + 1) >= 3 * v25 )
        goto LABEL_52;
      v27 = v25 >> 3;
      if ( v25 - *(_DWORD *)(a1 + 988) - v43 <= (unsigned int)v27 )
      {
        sub_1059000(v46, v25);
        goto LABEL_53;
      }
LABEL_48:
      *(_DWORD *)(a1 + 984) = v43;
      if ( *v29 != -4096 )
        --*(_DWORD *)(a1 + 988);
      *v29 = v20;
      v33 = (__int64 **)(v29 + 1);
      v29[1] = 0;
LABEL_14:
      *v33 = v26;
      if ( *(_BYTE *)(a1 + 132) )
      {
        v34 = *(_QWORD **)(a1 + 112);
        v35 = &v34[*(unsigned int *)(a1 + 124)];
        if ( v34 == v35 )
          goto LABEL_20;
        while ( v51 != *v34 )
        {
          if ( v35 == ++v34 )
            goto LABEL_20;
        }
      }
      else if ( !sub_C8CA60(a1 + 104, v51) )
      {
        goto LABEL_20;
      }
      sub_26C2C80((__int64)v52, a1 + 104, v50, v20, (__int64)&v51, v27);
LABEL_20:
      v36 = *sub_26CC460(v47, &v51);
      if ( v48 >= v36 )
        v36 = v48;
      v48 = v36;
LABEL_23:
      if ( v9 == ++v6 )
      {
        v7 = v47;
        goto LABEL_25;
      }
    }
    ++*(_QWORD *)(a1 + 968);
    v52[0] = 0;
LABEL_52:
    sub_1059000(v46, 2 * v25);
LABEL_53:
    sub_26CE030(v46, &v51, v52);
    v20 = v51;
    v29 = (__int64 *)v52[0];
    v43 = *(_DWORD *)(a1 + 984) + 1;
    goto LABEL_48;
  }
LABEL_25:
  v37 = *(_QWORD *)(v50[9] + 80);
  if ( v37 && v50 == (__int64 *)(v37 - 24) )
  {
    v41 = *(_QWORD *)(*(_QWORD *)(a1 + 1200) + 64LL);
    result = sub_26CC460(v7, (__int64 *)&v50);
    *result = v41 + 1;
  }
  else
  {
    result = sub_26CC460(v7, (__int64 *)&v50);
    *result = v48;
  }
  return result;
}
