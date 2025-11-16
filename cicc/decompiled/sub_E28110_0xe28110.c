// Function: sub_E28110
// Address: 0xe28110
//
unsigned __int64 __fastcall sub_E28110(__int64 a1, __int64 *a2, _BYTE *a3)
{
  _BYTE *v4; // rdx
  _QWORD *v5; // rdx
  _QWORD *v6; // rax
  _QWORD *v7; // rdx
  bool v8; // zf
  __int64 v9; // r15
  _QWORD *v10; // r13
  __int64 v11; // r14
  _BYTE *v12; // rsi
  int v13; // eax
  signed int v14; // eax
  _QWORD *v15; // rdx
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rdx
  unsigned __int64 result; // rax
  __int64 v21; // rdx
  _QWORD *v22; // rcx
  _QWORD *v23; // rax
  _BYTE *v24; // rdx
  __int64 v25; // rcx
  _QWORD *v26; // rax
  __int64 v27; // rax
  __int64 *v28; // rax
  __int64 *v29; // r14
  __int64 v30; // rax
  __int64 v31; // rax
  _QWORD *v32; // rax
  _QWORD *v33; // r14
  __int64 v34; // rsi
  _QWORD *v35; // [rsp+8h] [rbp-58h]
  __int64 v36; // [rsp+8h] [rbp-58h]
  _QWORD *v38; // [rsp+28h] [rbp-38h] BYREF

  if ( *a2 )
  {
    v4 = (_BYTE *)a2[1];
    if ( *v4 == 88 )
    {
      --*a2;
      a2[1] = (__int64)(v4 + 1);
      return 0;
    }
  }
  v5 = *(_QWORD **)(a1 + 16);
  v6 = (_QWORD *)((*v5 + v5[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
  v5[1] = (char *)v6 - *v5 + 16;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v28 = (__int64 *)sub_22077B0(32);
    v29 = v28;
    if ( v28 )
    {
      *v28 = 0;
      v28[1] = 0;
      v28[2] = 0;
      v28[3] = 0;
    }
    v30 = sub_2207820(4096);
    v29[2] = 4096;
    *v29 = v30;
    v7 = (_QWORD *)v30;
    v31 = *(_QWORD *)(a1 + 16);
    v29[1] = 16;
    v29[3] = v31;
    *(_QWORD *)(a1 + 16) = v29;
    if ( v7 )
    {
      *v7 = 0;
      v7[1] = 0;
    }
  }
  else
  {
    v7 = 0;
    if ( v6 )
    {
      *v6 = 0;
      v7 = v6;
      v6[1] = 0;
    }
  }
  v8 = *(_BYTE *)(a1 + 8) == 0;
  v38 = v7;
  if ( !v8 )
    return 0;
  v9 = 1;
  v10 = &v38;
  while ( 1 )
  {
    v11 = *a2;
    if ( !*a2 )
    {
LABEL_12:
      v15 = *(_QWORD **)(a1 + 16);
      v16 = (_QWORD *)((*v15 + v15[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
      v15[1] = (char *)v16 + 16LL - *v15;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        v26 = (_QWORD *)sub_22077B0(32);
        if ( v26 )
        {
          *v26 = 0;
          v26[1] = 0;
          v26[2] = 0;
          v26[3] = 0;
        }
        v35 = v26;
        v17 = (_QWORD *)sub_2207820(4096);
        *v35 = v17;
        v27 = *(_QWORD *)(a1 + 16);
        v35[2] = 4096;
        v35[3] = v27;
        *(_QWORD *)(a1 + 16) = v35;
        v35[1] = 16;
        if ( v17 )
        {
          *v17 = 0;
          v17[1] = 0;
        }
      }
      else
      {
        v17 = 0;
        if ( v16 )
        {
          *v16 = 0;
          v17 = v16;
          v16[1] = 0;
        }
      }
      *v10 = v17;
      v18 = sub_E27700(a1, a2, 0);
      if ( !v18 || *(_BYTE *)(a1 + 8) )
        return 0;
      *(_QWORD *)*v10 = v18;
      v19 = *(_QWORD *)(a1 + 104);
      if ( (unsigned __int64)(v11 - *a2) > 1 && v19 <= 9 )
      {
        *(_QWORD *)(a1 + 104) = v19 + 1;
        *(_QWORD *)(a1 + 8 * v19 + 24) = v18;
      }
      v10 = (_QWORD *)(*v10 + 8LL);
      goto LABEL_21;
    }
    v12 = (_BYTE *)a2[1];
    v13 = (char)*v12;
    if ( *v12 == 64 || (_BYTE)v13 == 90 )
      break;
    v14 = v13 - 48;
    if ( (unsigned int)v14 > 9 )
      goto LABEL_12;
    v21 = v14;
    if ( *(_QWORD *)(a1 + 104) <= (unsigned __int64)v14 )
    {
      *(_BYTE *)(a1 + 8) = 1;
      return 0;
    }
    a2[1] = (__int64)(v12 + 1);
    *a2 = v11 - 1;
    v22 = *(_QWORD **)(a1 + 16);
    v23 = (_QWORD *)((*v22 + v22[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
    v22[1] = (char *)v23 + 16LL - *v22;
    if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
    {
      v36 = v21;
      v32 = (_QWORD *)sub_22077B0(32);
      v33 = v32;
      if ( v32 )
      {
        *v32 = 0;
        v32[1] = 0;
        v32[2] = 0;
        v32[3] = 0;
      }
      v23 = (_QWORD *)sub_2207820(4096);
      v34 = *(_QWORD *)(a1 + 16);
      v21 = v36;
      v33[2] = 4096;
      *v33 = v23;
      v33[3] = v34;
      *(_QWORD *)(a1 + 16) = v33;
      v33[1] = 16;
      if ( !v23 )
      {
LABEL_43:
        *v10 = 0;
        MEMORY[0] = *(_QWORD *)(a1 + 8 * v21 + 24);
        BUG();
      }
    }
    else if ( !v23 )
    {
      goto LABEL_43;
    }
    *v23 = 0;
    v23[1] = 0;
    *v10 = v23;
    *v23 = *(_QWORD *)(a1 + 8 * v21 + 24);
    v10 = (_QWORD *)(*v10 + 8LL);
LABEL_21:
    ++v9;
    if ( *(_BYTE *)(a1 + 8) )
      return 0;
  }
  result = sub_E208B0((__int64 **)(a1 + 16), v38, v9 - 1);
  v24 = (_BYTE *)a2[1];
  v25 = *a2;
  if ( *v24 == 64 )
  {
    a2[1] = (__int64)(v24 + 1);
    *a2 = v25 - 1;
  }
  else
  {
    *a2 = v25 - 1;
    a2[1] = (__int64)(v24 + 1);
    *a3 = 1;
  }
  return result;
}
