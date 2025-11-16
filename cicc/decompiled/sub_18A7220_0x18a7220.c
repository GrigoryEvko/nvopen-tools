// Function: sub_18A7220
// Address: 0x18a7220
//
__int64 __fastcall sub_18A7220(__int64 a1, __int64 a2)
{
  const char *v3; // rax
  size_t v4; // rdx
  unsigned __int64 v5; // rax
  size_t v6; // rsi
  int *v7; // rdi
  int v8; // edx
  int *v9; // r14
  size_t v10; // rdx
  size_t v11; // r13
  int v12; // eax
  unsigned int v13; // r9d
  _QWORD *v14; // r10
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v18; // rax
  unsigned int v19; // r9d
  _QWORD *v20; // r10
  _QWORD *v21; // r11
  _BYTE *v22; // rcx
  __int64 *v23; // rdx
  __int64 v24; // rax
  _BYTE *v25; // rax
  _QWORD *v26; // [rsp+0h] [rbp-80h]
  _QWORD *v27; // [rsp+0h] [rbp-80h]
  unsigned int v28; // [rsp+8h] [rbp-78h]
  _QWORD *v29; // [rsp+8h] [rbp-78h]
  unsigned int v30; // [rsp+8h] [rbp-78h]
  unsigned int v31; // [rsp+10h] [rbp-70h]
  _QWORD *v32; // [rsp+10h] [rbp-70h]
  __int64 v33; // [rsp+18h] [rbp-68h]
  _QWORD *v34; // [rsp+18h] [rbp-68h]
  char v35; // [rsp+2Fh] [rbp-51h] BYREF
  int *v36; // [rsp+30h] [rbp-50h] BYREF
  size_t v37; // [rsp+38h] [rbp-48h]
  _QWORD v38[8]; // [rsp+40h] [rbp-40h] BYREF

  v3 = sub_1649960(a2);
  v37 = v4;
  v36 = (int *)v3;
  v35 = 46;
  v5 = sub_16D20C0((__int64 *)&v36, &v35, 1u, 0);
  if ( v5 == -1 )
  {
    v7 = v36;
    v6 = v37;
  }
  else
  {
    v6 = v5;
    v7 = v36;
    if ( v5 && v5 > v37 )
      v6 = v37;
  }
  v8 = *(_DWORD *)(a1 + 64);
  LOBYTE(v38[0]) = 0;
  v36 = (int *)v38;
  v37 = 0;
  v9 = sub_18A5420(v7, v6, v8, &v36);
  v11 = v10;
  v33 = *(_QWORD *)(a1 + 8) + 8LL * *(unsigned int *)(a1 + 16);
  v12 = sub_16D1B30((__int64 *)(a1 + 8), (unsigned __int8 *)v9, v10);
  if ( v12 == -1 )
  {
    if ( v33 != *(_QWORD *)(a1 + 8) + 8LL * *(unsigned int *)(a1 + 16) )
    {
LABEL_5:
      v13 = sub_16D19C0(a1 + 8, (unsigned __int8 *)v9, v11);
      v14 = (_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL * v13);
      v15 = *v14;
      if ( *v14 )
      {
        if ( v15 != -8 )
        {
LABEL_7:
          v16 = v15 + 8;
          goto LABEL_8;
        }
        --*(_DWORD *)(a1 + 24);
      }
      v26 = v14;
      v28 = v13;
      v18 = malloc(v11 + 137);
      v19 = v28;
      v20 = v26;
      v21 = (_QWORD *)v18;
      if ( !v18 )
      {
        if ( v11 == -137 )
        {
          v24 = malloc(1u);
          v21 = 0;
          v19 = v28;
          v20 = v26;
          if ( v24 )
          {
            v22 = (_BYTE *)(v24 + 136);
            v21 = (_QWORD *)v24;
            goto LABEL_27;
          }
        }
        v27 = v20;
        v30 = v19;
        v32 = v21;
        sub_16BD1C0("Allocation failed", 1u);
        v21 = v32;
        v19 = v30;
        v20 = v27;
      }
      v22 = v21 + 17;
      if ( v11 + 1 <= 1 )
      {
LABEL_17:
        v22[v11] = 0;
        *v21 = v11;
        memset(v21 + 1, 0, 0x80u);
        v21[8] = v21 + 6;
        v21[9] = v21 + 6;
        v21[14] = v21 + 12;
        v21[15] = v21 + 12;
        *v20 = v21;
        ++*(_DWORD *)(a1 + 20);
        v23 = (__int64 *)(*(_QWORD *)(a1 + 8) + 8LL * (unsigned int)sub_16D1CD0(a1 + 8, v19));
        v15 = *v23;
        if ( *v23 == -8 || !v15 )
        {
          do
          {
            do
            {
              v15 = v23[1];
              ++v23;
            }
            while ( !v15 );
          }
          while ( v15 == -8 );
        }
        goto LABEL_7;
      }
LABEL_27:
      v29 = v20;
      v31 = v19;
      v34 = v21;
      v25 = memcpy(v22, v9, v11);
      v20 = v29;
      v19 = v31;
      v21 = v34;
      v22 = v25;
      goto LABEL_17;
    }
  }
  else if ( v33 != *(_QWORD *)(a1 + 8) + 8LL * v12 )
  {
    goto LABEL_5;
  }
  v16 = 0;
LABEL_8:
  if ( v36 != (int *)v38 )
    j_j___libc_free_0(v36, v38[0] + 1LL);
  return v16;
}
