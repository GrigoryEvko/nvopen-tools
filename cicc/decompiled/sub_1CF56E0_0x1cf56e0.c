// Function: sub_1CF56E0
// Address: 0x1cf56e0
//
__int64 __fastcall sub_1CF56E0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v5; // r15
  _QWORD *v9; // rdi
  unsigned __int8 v10; // al
  unsigned __int8 v11; // si
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r9
  __int64 v15; // r8
  unsigned int v16; // edi
  __int64 *v17; // rdx
  __int64 v18; // r10
  __int64 v19; // rdi
  __int64 v20; // rax
  unsigned __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // r8
  unsigned int v24; // ecx
  __int64 *v25; // rax
  __int64 v26; // r9
  int v27; // eax
  char v29; // si
  int v30; // edx
  int v31; // ecx
  int v32; // eax
  int v33; // r11d
  __int64 v34; // [rsp+8h] [rbp-78h]
  __int64 v35; // [rsp+10h] [rbp-70h]
  __int64 v36; // [rsp+18h] [rbp-68h]
  __m128i v37; // [rsp+20h] [rbp-60h] BYREF
  __int64 v38; // [rsp+30h] [rbp-50h]
  __int64 v39; // [rsp+38h] [rbp-48h]
  __int64 v40; // [rsp+40h] [rbp-40h]
  char v41; // [rsp+48h] [rbp-38h]

  switch ( *(_BYTE *)(a3 + 16) )
  {
    case '6':
      sub_141EB40(&v37, (__int64 *)a3);
      goto LABEL_3;
    case '7':
      sub_141EDF0(&v37, a3);
      goto LABEL_3;
    case ':':
      sub_141F110(&v37, a3);
      goto LABEL_3;
    case ';':
      sub_141F3C0(&v37, a3);
      goto LABEL_3;
    case 'R':
      sub_141F0A0(&v37, a3);
LABEL_3:
      v4 = v37.m128i_i64[1];
      v5 = v37.m128i_i64[0];
      v35 = v38;
      v34 = v39;
      v36 = v40;
      break;
    default:
      break;
  }
  v9 = (_QWORD *)a2[1];
  v41 = 0;
  v10 = sub_13575E0(v9, a3, &v37, a4);
  v11 = v10 & 1;
  if ( (v10 & 1) != 0 )
  {
    v12 = *a2;
    v29 = v10 >> 1;
    v13 = *(unsigned int *)(*a2 + 48LL);
    v11 = (v29 ^ 1) & 1;
    if ( !(_DWORD)v13 )
      goto LABEL_14;
  }
  else
  {
    v12 = *a2;
    v13 = *(unsigned int *)(*a2 + 48LL);
    if ( !(_DWORD)v13 )
      goto LABEL_14;
  }
  v14 = *(_QWORD *)(a3 + 40);
  v15 = *(_QWORD *)(v12 + 32);
  v16 = (v13 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
  v17 = (__int64 *)(v15 + 16LL * v16);
  v18 = *v17;
  if ( v14 != *v17 )
  {
    v30 = 1;
    while ( v18 != -8 )
    {
      v31 = v30 + 1;
      v16 = (v13 - 1) & (v30 + v16);
      v17 = (__int64 *)(v15 + 16LL * v16);
      v18 = *v17;
      if ( v14 == *v17 )
        goto LABEL_7;
      v30 = v31;
    }
    goto LABEL_14;
  }
LABEL_7:
  if ( v17 == (__int64 *)(v15 + 16 * v13) || (v19 = v17[1]) == 0 )
  {
LABEL_14:
    *(_BYTE *)(a1 + 64) = 0;
    return a1;
  }
  v20 = a2[2];
  v21 = a3 & 0xFFFFFFFFFFFFFFFBLL | (4LL * v11);
  v22 = *(unsigned int *)(v20 + 32);
  v23 = *(_QWORD *)(v20 + 16);
  if ( !(_DWORD)v22 )
  {
LABEL_15:
    v25 = (__int64 *)(v23 + 16 * v22);
    goto LABEL_11;
  }
  v24 = (v22 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v25 = (__int64 *)(v23 + 16LL * v24);
  v26 = *v25;
  if ( a3 != *v25 )
  {
    v32 = 1;
    while ( v26 != -8 )
    {
      v33 = v32 + 1;
      v24 = (v22 - 1) & (v32 + v24);
      v25 = (__int64 *)(v23 + 16LL * v24);
      v26 = *v25;
      if ( a3 == *v25 )
        goto LABEL_11;
      v32 = v33;
    }
    goto LABEL_15;
  }
LABEL_11:
  v27 = *((_DWORD *)v25 + 2);
  *(_BYTE *)(a1 + 64) = 1;
  *(_QWORD *)a1 = v21;
  *(_QWORD *)(a1 + 24) = v35;
  *(_QWORD *)(a1 + 8) = v5;
  *(_QWORD *)(a1 + 32) = v34;
  *(_QWORD *)(a1 + 16) = v4;
  *(_QWORD *)(a1 + 40) = v36;
  *(_QWORD *)(a1 + 48) = v19;
  *(_DWORD *)(a1 + 56) = v27;
  return a1;
}
