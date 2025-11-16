// Function: sub_2BFB120
// Address: 0x2bfb120
//
__int64 __fastcall sub_2BFB120(__int64 a1, __int64 a2, unsigned int *a3)
{
  unsigned int v5; // esi
  __int64 v6; // rdi
  __int64 v7; // rdx
  char v8; // r9
  __int64 v9; // r10
  unsigned int v10; // r13d
  unsigned int v11; // eax
  _QWORD *v12; // r8
  __int64 v13; // r11
  _QWORD *v14; // rcx
  __int64 v15; // r14
  int v16; // r14d
  __int64 *v17; // rcx
  _QWORD *v18; // rax
  __int64 v19; // r13
  unsigned int v20; // esi
  __int64 v21; // rcx
  __int64 v22; // rdi
  __int64 *v23; // r9
  int v24; // r14d
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // r10
  __int64 v28; // rax
  __int64 *v29; // r15
  unsigned __int8 *v30; // r14
  __int64 v31; // rdi
  __int64 (__fastcall *v32)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v33; // r12
  _QWORD *v35; // rax
  __int64 v36; // rbx
  __int64 v37; // r13
  __int64 v38; // rdx
  unsigned int v39; // esi
  int v40; // eax
  int v41; // edx
  int v42; // eax
  int v43; // edx
  __int64 v44; // r14
  unsigned int v45; // r15d
  int v46; // ecx
  int v47; // [rsp+4h] [rbp-9Ch]
  __int64 v48; // [rsp+8h] [rbp-98h] BYREF
  char v49[32]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v50; // [rsp+30h] [rbp-70h]
  _QWORD v51[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v52; // [rsp+60h] [rbp-40h]

  v48 = a2;
  if ( !sub_2BF04A0(a2) )
    return *(_QWORD *)(v48 + 40);
  v5 = *(_DWORD *)(a1 + 88);
  v6 = v48;
  v7 = *a3;
  v8 = *((_BYTE *)a3 + 4);
  v9 = *(_QWORD *)(a1 + 72);
  if ( !v5 )
    goto LABEL_14;
  v10 = v5 - 1;
  v11 = (v5 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
  v12 = (_QWORD *)(v9 + 56LL * v11);
  v13 = *v12;
  v14 = v12;
  if ( v48 != *v12 )
  {
    v44 = *v12;
    v45 = (v5 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
    v46 = 1;
    while ( v44 != -4096 )
    {
      v45 = v10 & (v46 + v45);
      v47 = v46 + 1;
      v14 = (_QWORD *)(v9 + 56LL * v45);
      v44 = *v14;
      if ( v48 == *v14 )
        goto LABEL_4;
      v46 = v47;
    }
    goto LABEL_14;
  }
LABEL_4:
  if ( v14 == (_QWORD *)(v9 + 56LL * v5) )
    goto LABEL_14;
  if ( v8 != 1 )
  {
    if ( (unsigned int)v7 < *((_DWORD *)v14 + 4) )
    {
      v15 = (unsigned int)v7;
      goto LABEL_8;
    }
LABEL_14:
    if ( !(_DWORD)v7 && !v8 )
      goto LABEL_17;
    goto LABEL_16;
  }
  v15 = (unsigned int)(*(_DWORD *)(a1 + 8) + v7);
  if ( (unsigned int)v15 >= *((_DWORD *)v14 + 4) )
  {
LABEL_16:
    if ( (unsigned __int8)sub_2AAA120(v48) && sub_2BEF470(a1, v48, 0, 0) )
      return *(_QWORD *)*sub_2BF2580(a1 + 64, &v48);
LABEL_17:
    v20 = *(_DWORD *)(a1 + 56);
    if ( v20 )
    {
      v21 = v48;
      v22 = *(_QWORD *)(a1 + 40);
      v23 = 0;
      v24 = 1;
      v25 = (v20 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
      v26 = (__int64 *)(v22 + 16LL * v25);
      v27 = *v26;
      if ( *v26 == v48 )
      {
LABEL_19:
        v19 = v26[1];
        goto LABEL_20;
      }
      while ( v27 != -4096 )
      {
        if ( !v23 && v27 == -8192 )
          v23 = v26;
        v25 = (v20 - 1) & (v24 + v25);
        v26 = (__int64 *)(v22 + 16LL * v25);
        v27 = *v26;
        if ( v48 == *v26 )
          goto LABEL_19;
        ++v24;
      }
      if ( !v23 )
        v23 = v26;
      v40 = *(_DWORD *)(a1 + 48);
      ++*(_QWORD *)(a1 + 32);
      v41 = v40 + 1;
      v51[0] = v23;
      if ( 4 * (v40 + 1) < 3 * v20 )
      {
        if ( v20 - *(_DWORD *)(a1 + 52) - v41 > v20 >> 3 )
        {
LABEL_48:
          *(_DWORD *)(a1 + 48) = v41;
          if ( *v23 != -4096 )
            --*(_DWORD *)(a1 + 52);
          *v23 = v21;
          v19 = 0;
          v23[1] = 0;
LABEL_20:
          if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v19 + 8) + 8LL) - 17 > 1 )
            return v19;
          v28 = sub_2BF0180(a3, *(_QWORD *)(a1 + 904), (__int64 *)(a1 + 8));
          v29 = *(__int64 **)(a1 + 904);
          v50 = 257;
          v30 = (unsigned __int8 *)v28;
          v31 = v29[10];
          v32 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int8 *))(*(_QWORD *)v31 + 96LL);
          if ( v32 == sub_948070 )
          {
            if ( *(_BYTE *)v19 > 0x15u || *v30 > 0x15u )
              goto LABEL_33;
            v33 = sub_AD5840(v19, v30, 0);
          }
          else
          {
            v33 = v32(v31, (_BYTE *)v19, v30);
          }
          if ( v33 )
            return v33;
LABEL_33:
          v52 = 257;
          v35 = sub_BD2C40(72, 2u);
          v33 = (__int64)v35;
          if ( v35 )
            sub_B4DE80((__int64)v35, v19, (__int64)v30, (__int64)v51, 0, 0);
          (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)v29[11] + 16LL))(
            v29[11],
            v33,
            v49,
            v29[7],
            v29[8]);
          v36 = *v29;
          v37 = *v29 + 16LL * *((unsigned int *)v29 + 2);
          if ( *v29 != v37 )
          {
            do
            {
              v38 = *(_QWORD *)(v36 + 8);
              v39 = *(_DWORD *)v36;
              v36 += 16;
              sub_B99FD0(v33, v39, v38);
            }
            while ( v37 != v36 );
          }
          return v33;
        }
LABEL_71:
        sub_2AC6AB0(a1 + 32, v20);
        sub_2ABE290(a1 + 32, &v48, v51);
        v21 = v48;
        v23 = (__int64 *)v51[0];
        v41 = *(_DWORD *)(a1 + 48) + 1;
        goto LABEL_48;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 32);
      v51[0] = 0;
    }
    v20 *= 2;
    goto LABEL_71;
  }
LABEL_8:
  if ( !*(_QWORD *)(v14[1] + 8 * v15) )
    goto LABEL_14;
  v16 = 1;
  v17 = 0;
  if ( v48 != v13 )
  {
    while ( v13 != -4096 )
    {
      if ( !v17 && v13 == -8192 )
        v17 = v12;
      v11 = v10 & (v16 + v11);
      v12 = (_QWORD *)(v9 + 56LL * v11);
      v13 = *v12;
      if ( v48 == *v12 )
        goto LABEL_10;
      ++v16;
    }
    v42 = *(_DWORD *)(a1 + 80);
    if ( !v17 )
      v17 = v12;
    ++*(_QWORD *)(a1 + 64);
    v43 = v42 + 1;
    v51[0] = v17;
    if ( 4 * (v42 + 1) >= 3 * v5 )
    {
      v5 *= 2;
    }
    else if ( v5 - *(_DWORD *)(a1 + 84) - v43 > v5 >> 3 )
    {
LABEL_61:
      *(_DWORD *)(a1 + 80) = v43;
      if ( *v17 != -4096 )
        --*(_DWORD *)(a1 + 84);
      v18 = v17 + 3;
      *v17 = v6;
      v17[1] = (__int64)(v17 + 3);
      v17[2] = 0x400000000LL;
      v8 = *((_BYTE *)a3 + 4);
      v7 = *a3;
      goto LABEL_11;
    }
    sub_2AC6C60(a1 + 64, v5);
    sub_2ABE350(a1 + 64, &v48, v51);
    v6 = v48;
    v17 = (__int64 *)v51[0];
    v43 = *(_DWORD *)(a1 + 80) + 1;
    goto LABEL_61;
  }
LABEL_10:
  v18 = (_QWORD *)v12[1];
LABEL_11:
  if ( v8 == 1 )
    v7 = (unsigned int)(*(_DWORD *)(a1 + 8) + v7);
  return v18[v7];
}
