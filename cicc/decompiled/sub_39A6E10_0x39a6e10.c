// Function: sub_39A6E10
// Address: 0x39a6e10
//
__int64 __fastcall sub_39A6E10(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // rcx
  __int64 v7; // r13
  unsigned __int8 *v8; // r14
  int v9; // r14d
  unsigned int v10; // eax
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // rdx
  _BYTE *v14; // rdi
  size_t v15; // rdx
  size_t v16; // rcx
  __int64 result; // rax
  __int64 v18; // rdx
  int v19; // eax
  __int64 v20; // rsi
  int v21; // r8d
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // r10
  __int64 v25; // rdi
  __int64 v26; // rdx
  int v27; // eax
  int v28; // r11d
  unsigned __int8 *v29; // [rsp+0h] [rbp-50h]
  __int64 v30; // [rsp+8h] [rbp-48h]
  _BYTE v31[2]; // [rsp+1Ch] [rbp-34h] BYREF
  char v32; // [rsp+1Eh] [rbp-32h]

  v5 = a2;
  v6 = *(unsigned int *)(a2 + 8);
  v7 = *(_QWORD *)(a2 + 8 * (6 - v6));
  if ( v7 )
  {
    v8 = *(unsigned __int8 **)(a2 + 8 * (6 - v6));
    v30 = 0;
    v29 = sub_39A23D0((__int64)a1, v8);
    if ( *(_BYTE *)(a1[25] + 4498) )
    {
      v25 = *(_QWORD *)(v7 + 8 * (3LL - *(unsigned int *)(v7 + 8)));
      if ( v25 )
      {
        sub_161E970(v25);
        v30 = v26;
      }
    }
    if ( *(_BYTE *)v7 != 15 )
      v8 = *(unsigned __int8 **)(v7 - 8LL * *(unsigned int *)(v7 + 8));
    v9 = (*(__int64 (__fastcall **)(__int64 *, unsigned __int8 *))(*a1 + 48))(a1, v8);
    if ( *(_BYTE *)a2 != 15 )
      a2 = *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8));
    v10 = (*(__int64 (__fastcall **)(__int64 *, __int64))(*a1 + 48))(a1, a2);
    if ( v9 == v10 )
    {
      v11 = *(unsigned int *)(v5 + 24);
      if ( *(_DWORD *)(v7 + 24) == (_DWORD)v11 )
      {
LABEL_9:
        v12 = *(unsigned int *)(v5 + 8);
        goto LABEL_10;
      }
    }
    else
    {
      v32 = 0;
      sub_39A3560((__int64)a1, (__int64 *)(a3 + 8), 58, (__int64)v31, v10);
      v11 = *(unsigned int *)(v5 + 24);
      if ( *(_DWORD *)(v7 + 24) == (_DWORD)v11 )
        goto LABEL_9;
    }
    v32 = 0;
    sub_39A3560((__int64)a1, (__int64 *)(a3 + 8), 59, (__int64)v31, v11);
    goto LABEL_9;
  }
  v29 = 0;
  v12 = *(unsigned int *)(a2 + 8);
  v30 = 0;
LABEL_10:
  v13 = 0;
  if ( (unsigned int)v12 > 9 )
    v13 = *(_QWORD *)(v5 + 8 * (9 - v12));
  sub_39A6D90(a1, a3, v13);
  v14 = *(_BYTE **)(v5 + 8 * (3LL - *(unsigned int *)(v5 + 8)));
  if ( v14 )
  {
    v14 = (_BYTE *)sub_161E970((__int64)v14);
    v16 = v15;
  }
  else
  {
    v16 = 0;
  }
  if ( !v30 )
  {
    if ( *(_BYTE *)(a1[25] + 4498) )
    {
LABEL_16:
      sub_39A40D0(a1, a3, v14, v16);
      goto LABEL_17;
    }
    v18 = a1[26];
    v19 = *(_DWORD *)(v18 + 320);
    if ( v19 )
    {
      v20 = *(_QWORD *)(v18 + 304);
      v21 = v19 - 1;
      v22 = (v19 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v23 = (__int64 *)(v20 + 16LL * v22);
      v24 = *v23;
      if ( v5 == *v23 )
      {
LABEL_25:
        if ( v23[1] )
          goto LABEL_16;
      }
      else
      {
        v27 = 1;
        while ( v24 != -8 )
        {
          v28 = v27 + 1;
          v22 = v21 & (v27 + v22);
          v23 = (__int64 *)(v20 + 16LL * v22);
          v24 = *v23;
          if ( v5 == *v23 )
            goto LABEL_25;
          v27 = v28;
        }
      }
    }
  }
LABEL_17:
  result = 0;
  if ( v29 )
  {
    sub_39A3B20((__int64)a1, a3, 71, (__int64)v29);
    return 1;
  }
  return result;
}
