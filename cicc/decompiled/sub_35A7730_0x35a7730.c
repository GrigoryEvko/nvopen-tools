// Function: sub_35A7730
// Address: 0x35a7730
//
__int64 __fastcall sub_35A7730(__int64 a1)
{
  unsigned int v1; // r13d
  _QWORD *v2; // r14
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // r13
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v18; // rsi
  unsigned int v19; // edx
  int v20; // esi
  int v21; // r9d
  int v22; // esi
  _DWORD *v23; // rdx
  int v24; // eax
  int v25; // [rsp+10h] [rbp-70h] BYREF
  int v26; // [rsp+14h] [rbp-6Ch] BYREF
  __int64 v27; // [rsp+18h] [rbp-68h] BYREF
  _DWORD *v28; // [rsp+20h] [rbp-60h] BYREF
  _DWORD *v29; // [rsp+28h] [rbp-58h] BYREF
  __int64 v30; // [rsp+30h] [rbp-50h] BYREF
  __int64 v31; // [rsp+38h] [rbp-48h]
  __int64 v32; // [rsp+40h] [rbp-40h]
  __int64 v33; // [rsp+48h] [rbp-38h]

  v1 = 0;
  if ( !sub_2EA48B0(a1) )
    return v1;
  v30 = 0;
  v2 = sub_2EA6400(a1);
  v3 = v2[4];
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v4 = *(_QWORD *)(v3 + 32);
  v5 = sub_2E311E0((__int64)v2);
  v6 = v2[7];
  v7 = v5;
  v27 = v6;
  if ( v6 == v5 )
  {
LABEL_36:
    v15 = (unsigned int)v33;
    v16 = v31;
    v1 = 1;
    goto LABEL_17;
  }
  while ( 1 )
  {
    v8 = *(_QWORD *)(v6 + 32);
    v9 = sub_2E88FE0(v6);
    v10 = *(_QWORD *)(v6 + 32);
    v11 = v8 + 40LL * v9;
    while ( v11 != v10 )
    {
      if ( !*(_BYTE *)v10 )
      {
        v12 = *(unsigned int *)(v10 + 8);
        v13 = (int)v12 < 0
            ? *(_QWORD *)(*(_QWORD *)(v4 + 56) + 16 * (v12 & 0x7FFFFFFF) + 8)
            : *(_QWORD *)(*(_QWORD *)(v4 + 304) + 8 * v12);
        if ( v13 )
        {
          if ( (*(_BYTE *)(v13 + 3) & 0x10) != 0 )
          {
            while ( 1 )
            {
              v13 = *(_QWORD *)(v13 + 32);
              if ( !v13 )
                break;
              if ( (*(_BYTE *)(v13 + 3) & 0x10) == 0 )
                goto LABEL_11;
            }
          }
          else
          {
LABEL_11:
            v14 = *(_QWORD *)(v13 + 16);
LABEL_12:
            if ( v2 != *(_QWORD **)(v14 + 24) || *(_WORD *)(v14 + 68) == 68 || !*(_WORD *)(v14 + 68) )
              goto LABEL_15;
            while ( 1 )
            {
              v13 = *(_QWORD *)(v13 + 32);
              if ( !v13 )
                break;
              if ( (*(_BYTE *)(v13 + 3) & 0x10) == 0 && v14 != *(_QWORD *)(v13 + 16) )
              {
                v14 = *(_QWORD *)(v13 + 16);
                goto LABEL_12;
              }
            }
          }
        }
      }
      v10 += 40;
    }
    v18 = *(_QWORD *)(v6 + 24);
    v25 = 0;
    v26 = 0;
    sub_3598310(v6, v18, &v25, &v26);
    if ( v26 >= 0 || v2 != *(_QWORD **)(sub_2EBEE10(v4, v26) + 24) )
      break;
    v15 = (unsigned int)v33;
    v16 = v31;
    if ( (_DWORD)v33 )
    {
      v19 = (v33 - 1) & (37 * v26);
      v20 = *(_DWORD *)(v31 + 4LL * v19);
      if ( v26 == v20 )
        goto LABEL_16;
      v21 = 1;
      while ( v20 != -1 )
      {
        v19 = (v33 - 1) & (v21 + v19);
        v20 = *(_DWORD *)(v31 + 4LL * v19);
        if ( v26 == v20 )
          goto LABEL_16;
        ++v21;
      }
    }
    if ( !(unsigned __int8)sub_3549AD0((__int64)&v30, &v26, &v28) )
    {
      v22 = v33;
      v23 = v28;
      ++v30;
      v24 = v32 + 1;
      v29 = v28;
      if ( 4 * ((int)v32 + 1) >= (unsigned int)(3 * v33) )
      {
        v22 = 2 * v33;
      }
      else if ( (int)v33 - HIDWORD(v32) - v24 > (unsigned int)v33 >> 3 )
      {
LABEL_39:
        LODWORD(v32) = v24;
        if ( *v23 != -1 )
          --HIDWORD(v32);
        *v23 = v26;
        goto LABEL_35;
      }
      sub_2E29BA0((__int64)&v30, v22);
      sub_3549AD0((__int64)&v30, &v26, &v29);
      v23 = v29;
      v24 = v32 + 1;
      goto LABEL_39;
    }
LABEL_35:
    sub_2FD79B0(&v27);
    v6 = v27;
    if ( v27 == v7 )
      goto LABEL_36;
  }
LABEL_15:
  v15 = (unsigned int)v33;
  v16 = v31;
LABEL_16:
  v1 = 0;
LABEL_17:
  sub_C7D6A0(v16, 4 * v15, 4);
  return v1;
}
