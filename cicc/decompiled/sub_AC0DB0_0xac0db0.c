// Function: sub_AC0DB0
// Address: 0xac0db0
//
__int64 __fastcall sub_AC0DB0(unsigned __int64 *a1, __int64 a2)
{
  __int64 v4; // r14
  unsigned __int64 v5; // r15
  __int64 v6; // r12
  unsigned __int64 v7; // rbx
  bool v8; // cc
  __int64 v9; // rdi
  __int64 result; // rax
  __int64 v11; // rdx
  __int64 v12; // r12
  unsigned __int64 v13; // rdi
  __int64 v14; // rsi
  int v15; // eax
  unsigned __int64 v16; // r12
  unsigned int v17; // eax
  unsigned int v18; // eax
  unsigned __int64 v19; // r12
  unsigned __int64 v20; // r15
  __int64 v21; // [rsp+8h] [rbp-78h]
  __int64 v22; // [rsp+8h] [rbp-78h]
  __int64 v23; // [rsp+8h] [rbp-78h]
  __int64 v24; // [rsp+8h] [rbp-78h]
  __int64 v25; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v26; // [rsp+18h] [rbp-68h]
  __int64 v27; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v28; // [rsp+28h] [rbp-58h]
  __int64 v29; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v30; // [rsp+38h] [rbp-48h]
  __int64 v31; // [rsp+40h] [rbp-40h]
  unsigned int v32; // [rsp+48h] [rbp-38h]

  v4 = a2 + 16;
  v5 = *a1;
  v6 = *a1 + 16;
  if ( (int)sub_C4C880(v6, a2) >= 0 )
  {
    if ( (int)sub_C4C880(v6, a2 + 16) <= 0 )
      v6 = a2 + 16;
    v28 = *(_DWORD *)(v6 + 8);
    if ( v28 > 0x40 )
    {
      sub_C43780(&v27, v6);
      v5 = *a1;
      v26 = *(_DWORD *)(*a1 + 8);
      if ( v26 <= 0x40 )
        goto LABEL_6;
    }
    else
    {
      v27 = *(_QWORD *)v6;
      v26 = *(_DWORD *)(v5 + 8);
      if ( v26 <= 0x40 )
      {
LABEL_6:
        v25 = *(_QWORD *)v5;
LABEL_7:
        sub_AADC30((__int64)&v29, (__int64)&v25, &v27);
        v7 = *a1;
        if ( *(_DWORD *)(*a1 + 8) > 0x40u && *(_QWORD *)v7 )
          j_j___libc_free_0_0(*(_QWORD *)v7);
        v8 = *(_DWORD *)(v7 + 24) <= 0x40u;
        *(_QWORD *)v7 = v29;
        *(_DWORD *)(v7 + 8) = v30;
        v30 = 0;
        if ( v8 || (v9 = *(_QWORD *)(v7 + 16)) == 0 )
        {
          *(_QWORD *)(v7 + 16) = v31;
          result = v32;
          *(_DWORD *)(v7 + 24) = v32;
        }
        else
        {
          j_j___libc_free_0_0(v9);
          v8 = v30 <= 0x40;
          *(_QWORD *)(v7 + 16) = v31;
          result = v32;
          *(_DWORD *)(v7 + 24) = v32;
          if ( !v8 && v29 )
          {
            result = j_j___libc_free_0_0(v29);
            if ( v26 <= 0x40 )
              goto LABEL_15;
            goto LABEL_23;
          }
        }
        if ( v26 <= 0x40 )
        {
LABEL_15:
          if ( v28 > 0x40 )
          {
            if ( v27 )
              return j_j___libc_free_0_0(v27);
          }
          return result;
        }
LABEL_23:
        if ( v25 )
          result = j_j___libc_free_0_0(v25);
        goto LABEL_15;
      }
    }
    sub_C43780(&v25, v5);
    goto LABEL_7;
  }
  v11 = a1[1];
  v12 = *(unsigned int *)(v11 + 8);
  v13 = *(_QWORD *)v11;
  v14 = v12 + 1;
  v15 = *(_DWORD *)(v11 + 8);
  if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(v11 + 12) )
  {
    if ( v13 > v5 || v5 >= v13 + 32 * v12 )
    {
      v24 = a1[1];
      sub_9D5330(v24, v14);
      v11 = v24;
      v12 = *(unsigned int *)(v24 + 8);
      v13 = *(_QWORD *)v24;
      v15 = *(_DWORD *)(v24 + 8);
    }
    else
    {
      v20 = v5 - v13;
      v23 = a1[1];
      sub_9D5330(v23, v14);
      v11 = v23;
      v13 = *(_QWORD *)v23;
      v12 = *(unsigned int *)(v23 + 8);
      v5 = *(_QWORD *)v23 + v20;
      v15 = *(_DWORD *)(v23 + 8);
    }
  }
  v16 = v13 + 32 * v12;
  if ( v16 )
  {
    v17 = *(_DWORD *)(v5 + 8);
    *(_DWORD *)(v16 + 8) = v17;
    if ( v17 > 0x40 )
    {
      v21 = v11;
      sub_C43780(v16, v5);
      v11 = v21;
    }
    else
    {
      *(_QWORD *)v16 = *(_QWORD *)v5;
    }
    v18 = *(_DWORD *)(v5 + 24);
    *(_DWORD *)(v16 + 24) = v18;
    if ( v18 > 0x40 )
    {
      v22 = v11;
      sub_C43780(v16 + 16, v5 + 16);
      v11 = v22;
      v15 = *(_DWORD *)(v22 + 8);
    }
    else
    {
      *(_QWORD *)(v16 + 16) = *(_QWORD *)(v5 + 16);
      v15 = *(_DWORD *)(v11 + 8);
    }
  }
  *(_DWORD *)(v11 + 8) = v15 + 1;
  v19 = *a1;
  if ( *(_DWORD *)(*a1 + 8) <= 0x40u && *(_DWORD *)(a2 + 8) <= 0x40u )
  {
    *(_QWORD *)v19 = *(_QWORD *)a2;
    *(_DWORD *)(v19 + 8) = *(_DWORD *)(a2 + 8);
  }
  else
  {
    sub_C43990(*a1, a2);
  }
  if ( *(_DWORD *)(v19 + 24) > 0x40u || *(_DWORD *)(a2 + 24) > 0x40u )
    return sub_C43990(v19 + 16, v4);
  *(_QWORD *)(v19 + 16) = *(_QWORD *)(a2 + 16);
  result = *(unsigned int *)(a2 + 24);
  *(_DWORD *)(v19 + 24) = result;
  return result;
}
