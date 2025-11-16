// Function: sub_2A19560
// Address: 0x2a19560
//
__int64 __fastcall sub_2A19560(__int64 **a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 *v7; // r12
  __int64 *v8; // r15
  __int64 v9; // r13
  _QWORD *v10; // rax
  _QWORD *v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rdx
  _QWORD *v14; // r12
  __int64 v15; // rdx
  _QWORD *v16; // r15
  _QWORD *v18; // rax
  __int64 *v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 *v23; // rax
  __int64 v24; // rsi
  unsigned __int64 v25; // rax
  __int64 v26; // r14
  unsigned int v27; // r13d
  __int64 v28; // rsi
  _QWORD *v29; // rax
  _QWORD *v30; // rdx
  __int64 *v31; // rax
  __int64 v32; // rsi
  __int64 v35; // [rsp+10h] [rbp-40h]
  __int64 v36; // [rsp+18h] [rbp-38h]
  int v37; // [rsp+18h] [rbp-38h]

  v5 = *a1[1];
  v6 = sub_D47930(v5);
  v7 = a1[5];
  v8 = a1[4];
  v36 = v6;
  if ( v7 != v8 )
  {
    while ( 1 )
    {
      v9 = *v8;
      if ( *(_BYTE *)(v5 + 84) )
        break;
      if ( sub_C8CA60(v5 + 56, *v8) )
      {
LABEL_7:
        if ( v7 == ++v8 )
          goto LABEL_8;
      }
      else
      {
LABEL_21:
        if ( (unsigned __int8)sub_B19720(a4, v36, v9) )
        {
          if ( *(_BYTE *)(a3 + 28) )
          {
            v23 = *(__int64 **)(a3 + 8);
            v24 = *(unsigned int *)(a3 + 20);
            v19 = &v23[v24];
            if ( v23 != v19 )
            {
              while ( v9 != *v23 )
              {
                if ( v19 == ++v23 )
                  goto LABEL_53;
              }
              goto LABEL_7;
            }
LABEL_53:
            if ( (unsigned int)v24 >= *(_DWORD *)(a3 + 16) )
              goto LABEL_48;
            ++v8;
            *(_DWORD *)(a3 + 20) = v24 + 1;
            *v19 = v9;
            ++*(_QWORD *)a3;
            if ( v7 == v8 )
              goto LABEL_8;
          }
          else
          {
LABEL_48:
            ++v8;
            sub_C8CC70(a3, v9, (__int64)v19, a3, v21, v22);
            if ( v7 == v8 )
              goto LABEL_8;
          }
        }
        else
        {
          if ( !*(_BYTE *)(a2 + 28) )
            goto LABEL_50;
          v31 = *(__int64 **)(a2 + 8);
          v32 = *(unsigned int *)(a2 + 20);
          v19 = &v31[v32];
          if ( v31 != v19 )
          {
            while ( v9 != *v31 )
            {
              if ( v19 == ++v31 )
                goto LABEL_56;
            }
            goto LABEL_7;
          }
LABEL_56:
          if ( (unsigned int)v32 < *(_DWORD *)(a2 + 16) )
          {
            ++v8;
            *(_DWORD *)(a2 + 20) = v32 + 1;
            *v19 = v9;
            ++*(_QWORD *)a2;
            if ( v7 == v8 )
              goto LABEL_8;
          }
          else
          {
LABEL_50:
            ++v8;
            sub_C8CC70(a2, v9, (__int64)v19, v20, v21, v22);
            if ( v7 == v8 )
              goto LABEL_8;
          }
        }
      }
    }
    v10 = *(_QWORD **)(v5 + 64);
    v11 = &v10[*(unsigned int *)(v5 + 76)];
    if ( v10 != v11 )
    {
      while ( v9 != *v10 )
      {
        if ( v11 == ++v10 )
          goto LABEL_21;
      }
      goto LABEL_7;
    }
    goto LABEL_21;
  }
LABEL_8:
  v35 = sub_D4B130(v5);
  v12 = *(_QWORD **)(a2 + 8);
  if ( *(_BYTE *)(a2 + 28) )
    v13 = *(unsigned int *)(a2 + 20);
  else
    v13 = *(unsigned int *)(a2 + 16);
  v14 = &v12[v13];
  if ( v12 == v14 )
    return 1;
  while ( 1 )
  {
    v15 = *v12;
    v16 = v12;
    if ( *v12 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v14 == ++v12 )
      return 1;
  }
  while ( 1 )
  {
    if ( v14 == v16 )
      return 1;
    if ( v35 != v15 )
    {
      v25 = *(_QWORD *)(v15 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v25 != v15 + 48 )
      {
        if ( !v25 )
          BUG();
        v26 = v25 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v25 - 24) - 30 <= 0xA )
        {
          v37 = sub_B46E30(v26);
          if ( v37 )
            break;
        }
      }
    }
LABEL_16:
    v18 = v16 + 1;
    if ( v16 + 1 == v14 )
      return 1;
    while ( 1 )
    {
      v15 = *v18;
      v16 = v18;
      if ( *v18 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v14 == ++v18 )
        return 1;
    }
  }
  v27 = 0;
  while ( 1 )
  {
    v28 = sub_B46EC0(v26, v27);
    if ( !*(_BYTE *)(a2 + 28) )
      break;
    v29 = *(_QWORD **)(a2 + 8);
    v30 = &v29[*(unsigned int *)(a2 + 20)];
    if ( v29 == v30 )
      return 0;
    while ( v28 != *v29 )
    {
      if ( v30 == ++v29 )
        return 0;
    }
LABEL_38:
    if ( ++v27 == v37 )
      goto LABEL_16;
  }
  if ( sub_C8CA60(a2, v28) )
    goto LABEL_38;
  return 0;
}
