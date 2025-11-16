// Function: sub_D48E00
// Address: 0xd48e00
//
char __fastcall sub_D48E00(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  char *v4; // rax
  __int64 *v5; // r13
  __int64 *v6; // rbx
  __int64 v7; // r12
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 *v10; // r12
  int v11; // ecx
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r10
  __int64 v15; // rsi
  int v16; // ecx
  __int64 v17; // rdi
  int v18; // ecx
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // r10
  __int64 v22; // rsi
  int v23; // ecx
  __int64 v24; // rdi
  int v25; // ecx
  unsigned int v26; // edx
  __int64 *v27; // rax
  __int64 v28; // r10
  __int64 v29; // rsi
  int v30; // ecx
  __int64 v31; // rdi
  int v32; // ecx
  unsigned int v33; // edx
  __int64 *v34; // rax
  __int64 v35; // r10
  int v36; // ecx
  __int64 v37; // rsi
  char v38; // r9
  __int64 v39; // rdi
  char result; // al
  int v41; // eax
  int v42; // eax
  int v43; // eax
  int v44; // eax
  int v45; // r11d
  int v46; // r11d
  int v47; // r11d
  int v48; // r11d
  _BYTE v49[4]; // [rsp+Ch] [rbp-44h] BYREF
  __int64 v50; // [rsp+10h] [rbp-40h] BYREF
  __int64 v51; // [rsp+18h] [rbp-38h]
  char *v52; // [rsp+20h] [rbp-30h]

  v4 = v49;
  v5 = *(__int64 **)(a1 + 40);
  v6 = *(__int64 **)(a1 + 32);
  v49[0] = a4;
  v50 = a3;
  v51 = a2;
  v52 = v49;
  v7 = ((char *)v5 - (char *)v6) >> 5;
  v8 = v5 - v6;
  if ( v7 > 0 )
  {
    v9 = a2;
    v10 = &v6[4 * v7];
    while ( 1 )
    {
      v36 = *(_DWORD *)(a3 + 24);
      v37 = *v6;
      v38 = *v4;
      v39 = *(_QWORD *)(a3 + 8);
      if ( v36 )
      {
        v11 = v36 - 1;
        v12 = v11 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
        v13 = (__int64 *)(v39 + 16LL * v12);
        v14 = *v13;
        if ( v37 == *v13 )
        {
LABEL_4:
          if ( !(unsigned __int8)sub_D46840(v13[1], v37, v9, v38) )
            return v5 == v6;
          goto LABEL_5;
        }
        v44 = 1;
        while ( v14 != -4096 )
        {
          v45 = v44 + 1;
          v12 = v11 & (v44 + v12);
          v13 = (__int64 *)(v39 + 16LL * v12);
          v14 = *v13;
          if ( v37 == *v13 )
            goto LABEL_4;
          v44 = v45;
        }
      }
      if ( !(unsigned __int8)sub_D46840(0, v37, v9, v38) )
        return v5 == v6;
LABEL_5:
      v15 = v6[1];
      v16 = *(_DWORD *)(v50 + 24);
      v17 = *(_QWORD *)(v50 + 8);
      if ( v16 )
      {
        v18 = v16 - 1;
        v19 = v18 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v20 = (__int64 *)(v17 + 16LL * v19);
        v21 = *v20;
        if ( v15 == *v20 )
        {
LABEL_7:
          if ( !(unsigned __int8)sub_D46840(v20[1], v15, v51, *v52) )
            return v5 == v6 + 1;
          goto LABEL_8;
        }
        v41 = 1;
        while ( v21 != -4096 )
        {
          v46 = v41 + 1;
          v19 = v18 & (v41 + v19);
          v20 = (__int64 *)(v17 + 16LL * v19);
          v21 = *v20;
          if ( v15 == *v20 )
            goto LABEL_7;
          v41 = v46;
        }
      }
      if ( !(unsigned __int8)sub_D46840(0, v15, v51, *v52) )
        return v5 == v6 + 1;
LABEL_8:
      v22 = v6[2];
      v23 = *(_DWORD *)(v50 + 24);
      v24 = *(_QWORD *)(v50 + 8);
      if ( v23 )
      {
        v25 = v23 - 1;
        v26 = v25 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v27 = (__int64 *)(v24 + 16LL * v26);
        v28 = *v27;
        if ( v22 == *v27 )
        {
LABEL_10:
          if ( !(unsigned __int8)sub_D46840(v27[1], v22, v51, *v52) )
            return v5 == v6 + 2;
          goto LABEL_11;
        }
        v42 = 1;
        while ( v28 != -4096 )
        {
          v47 = v42 + 1;
          v26 = v25 & (v42 + v26);
          v27 = (__int64 *)(v24 + 16LL * v26);
          v28 = *v27;
          if ( v22 == *v27 )
            goto LABEL_10;
          v42 = v47;
        }
      }
      if ( !(unsigned __int8)sub_D46840(0, v22, v51, *v52) )
        return v5 == v6 + 2;
LABEL_11:
      v29 = v6[3];
      v30 = *(_DWORD *)(v50 + 24);
      v31 = *(_QWORD *)(v50 + 8);
      if ( v30 )
      {
        v32 = v30 - 1;
        v33 = v32 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v34 = (__int64 *)(v31 + 16LL * v33);
        v35 = *v34;
        if ( v29 == *v34 )
        {
LABEL_13:
          if ( !(unsigned __int8)sub_D46840(v34[1], v29, v51, *v52) )
            return v5 == v6 + 3;
          goto LABEL_14;
        }
        v43 = 1;
        while ( v35 != -4096 )
        {
          v48 = v43 + 1;
          v33 = v32 & (v43 + v33);
          v34 = (__int64 *)(v31 + 16LL * v33);
          v35 = *v34;
          if ( v29 == *v34 )
            goto LABEL_13;
          v43 = v48;
        }
      }
      if ( !(unsigned __int8)sub_D46840(0, v29, v51, *v52) )
        return v5 == v6 + 3;
LABEL_14:
      v6 += 4;
      if ( v10 == v6 )
      {
        v8 = v5 - v6;
        break;
      }
      v4 = v52;
      v9 = v51;
      a3 = v50;
    }
  }
  if ( v8 != 2 )
  {
    if ( v8 != 3 )
    {
      result = 1;
      if ( v8 != 1 )
        return result;
      goto LABEL_40;
    }
    if ( !(unsigned __int8)sub_D46990(&v50, *v6) )
      return v5 == v6;
    ++v6;
  }
  if ( !(unsigned __int8)sub_D46990(&v50, *v6) )
    return v5 == v6;
  ++v6;
LABEL_40:
  result = sub_D46990(&v50, *v6);
  if ( !result )
    return v5 == v6;
  return result;
}
