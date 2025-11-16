// Function: sub_F11320
// Address: 0xf11320
//
__int64 __fastcall sub_F11320(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int16 v4; // cx
  __int64 v5; // rdi
  __int64 v6; // rsi
  __int64 v7; // r9
  __int64 v8; // r14
  unsigned __int64 v9; // rsi
  unsigned int *v10; // rax
  int v11; // ecx
  unsigned int *v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // r12
  __int64 v15; // r9
  __int64 v16; // r14
  unsigned __int64 v17; // rsi
  _QWORD *v18; // rax
  int v19; // ecx
  _QWORD *v20; // rdx
  __int64 v21; // rsi
  __int64 result; // rax
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rsi
  unsigned __int64 v26; // rsi
  _QWORD v27[5]; // [rsp+18h] [rbp-28h] BYREF

  v2 = *(_QWORD *)(a1 + 24);
  v3 = *(_QWORD *)a1;
  v4 = *(_WORD *)(a1 + 40);
  v5 = *(_QWORD *)(a1 + 32);
  if ( !v2 )
  {
    *(_QWORD *)(v3 + 48) = 0;
    *(_QWORD *)(v3 + 56) = 0;
    *(_WORD *)(v3 + 64) = 0;
    v13 = *(_QWORD *)(a1 + 48);
    v14 = *(_QWORD *)a1;
    v27[0] = v13;
    if ( !v13 )
      goto LABEL_28;
    goto LABEL_14;
  }
  *(_QWORD *)(v3 + 48) = v2;
  *(_QWORD *)(v3 + 56) = v5;
  *(_WORD *)(v3 + 64) = v4;
  if ( v5 != v2 + 48 )
  {
    if ( v5 )
      v5 -= 24;
    v6 = *(_QWORD *)sub_B46C60(v5);
    v27[0] = v6;
    if ( v6 && (sub_B96E90((__int64)v27, v6, 1), (v8 = v27[0]) != 0) )
    {
      v9 = *(unsigned int *)(v3 + 8);
      v10 = *(unsigned int **)v3;
      v11 = *(_DWORD *)(v3 + 8);
      v12 = (unsigned int *)(*(_QWORD *)v3 + 16 * v9);
      if ( *(unsigned int **)v3 != v12 )
      {
        while ( 1 )
        {
          v7 = *v10;
          if ( !(_DWORD)v7 )
            break;
          v10 += 4;
          if ( v12 == v10 )
            goto LABEL_38;
        }
        *((_QWORD *)v10 + 1) = v27[0];
        goto LABEL_12;
      }
LABEL_38:
      v24 = *(unsigned int *)(v3 + 12);
      if ( v9 >= v24 )
      {
        v26 = v9 + 1;
        if ( v24 < v26 )
        {
          sub_C8D5F0(v3, (const void *)(v3 + 16), v26, 0x10u, v3 + 16, v7);
          v12 = (unsigned int *)(*(_QWORD *)v3 + 16LL * *(unsigned int *)(v3 + 8));
        }
        *(_QWORD *)v12 = 0;
        *((_QWORD *)v12 + 1) = v8;
        ++*(_DWORD *)(v3 + 8);
        v8 = v27[0];
      }
      else
      {
        if ( v12 )
        {
          *v12 = 0;
          *((_QWORD *)v12 + 1) = v8;
          v11 = *(_DWORD *)(v3 + 8);
          v8 = v27[0];
        }
        *(_DWORD *)(v3 + 8) = v11 + 1;
      }
    }
    else
    {
      sub_93FB40(v3, 0);
      v8 = v27[0];
    }
    if ( v8 )
LABEL_12:
      sub_B91220((__int64)v27, v8);
  }
  v13 = *(_QWORD *)(a1 + 48);
  v14 = *(_QWORD *)a1;
  v27[0] = v13;
  if ( !v13 )
  {
LABEL_28:
    sub_93FB40(v14, 0);
    v16 = v27[0];
    goto LABEL_29;
  }
LABEL_14:
  sub_B96E90((__int64)v27, v13, 1);
  v16 = v27[0];
  if ( !v27[0] )
    goto LABEL_28;
  v17 = *(unsigned int *)(v14 + 8);
  v18 = *(_QWORD **)v14;
  v19 = *(_DWORD *)(v14 + 8);
  v20 = (_QWORD *)(*(_QWORD *)v14 + 16 * v17);
  if ( *(_QWORD **)v14 != v20 )
  {
    while ( *(_DWORD *)v18 )
    {
      v18 += 2;
      if ( v20 == v18 )
        goto LABEL_31;
    }
    v18[1] = v27[0];
    goto LABEL_20;
  }
LABEL_31:
  v23 = *(unsigned int *)(v14 + 12);
  if ( v17 >= v23 )
  {
    v25 = v17 + 1;
    if ( v23 < v25 )
    {
      sub_C8D5F0(v14, (const void *)(v14 + 16), v25, 0x10u, v14 + 16, v15);
      v20 = (_QWORD *)(*(_QWORD *)v14 + 16LL * *(unsigned int *)(v14 + 8));
    }
    *v20 = 0;
    v20[1] = v16;
    ++*(_DWORD *)(v14 + 8);
    v16 = v27[0];
  }
  else
  {
    if ( v20 )
    {
      *(_DWORD *)v20 = 0;
      v20[1] = v16;
      v19 = *(_DWORD *)(v14 + 8);
      v16 = v27[0];
    }
    *(_DWORD *)(v14 + 8) = v19 + 1;
  }
LABEL_29:
  if ( v16 )
LABEL_20:
    sub_B91220((__int64)v27, v16);
  v21 = *(_QWORD *)(a1 + 48);
  if ( v21 )
    sub_B91220(a1 + 48, v21);
  result = *(_QWORD *)(a1 + 24);
  if ( result != -4096 && result != 0 && result != -8192 )
    return sub_BD60C0((_QWORD *)(a1 + 8));
  return result;
}
