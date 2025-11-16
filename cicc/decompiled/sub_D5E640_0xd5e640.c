// Function: sub_D5E640
// Address: 0xd5e640
//
__int64 __fastcall sub_D5E640(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r13d
  unsigned int v5; // r14d
  unsigned __int8 v6; // al
  bool v7; // al
  const void **v8; // r15
  bool v9; // al
  unsigned int v10; // eax
  int v12; // eax
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // r13
  __int64 v17; // rbx
  unsigned int v18; // eax
  unsigned int v19; // eax
  bool v20; // al
  int v21; // eax
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rbx
  const void *v26; // rax
  const void *v27; // rax
  __int64 v28; // [rsp+0h] [rbp-60h]
  __int64 v29; // [rsp+0h] [rbp-60h]
  __int64 v30; // [rsp+0h] [rbp-60h]
  __int64 v31; // [rsp+0h] [rbp-60h]
  __int64 v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+8h] [rbp-58h]
  __int64 v35; // [rsp+8h] [rbp-58h]
  __int64 v36; // [rsp+8h] [rbp-58h]
  __int64 v37; // [rsp+8h] [rbp-58h]
  const void *v38; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v39; // [rsp+18h] [rbp-48h]
  const void *v40; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v41; // [rsp+28h] [rbp-38h]

  v4 = *(_DWORD *)(a3 + 8);
  if ( v4 <= 1 )
    goto LABEL_17;
  v5 = *(_DWORD *)(a3 + 24);
  if ( v5 <= 1 || *(_DWORD *)(a4 + 8) <= 1u || *(_DWORD *)(a4 + 24) <= 1u )
    goto LABEL_17;
  v6 = *(_BYTE *)(a2 + 16);
  if ( v6 == 2 )
  {
    v31 = a4;
    v36 = a3;
    v21 = sub_C4C880(a3, a4);
    v22 = v31;
    v23 = v36;
    v24 = v31;
    if ( v21 < 0 )
      v24 = v36;
    v39 = *(_DWORD *)(v24 + 8);
    if ( v39 > 0x40 )
    {
      sub_C43780((__int64)&v38, (const void **)v24);
      v22 = v31;
      v23 = v36;
    }
    else
    {
      v38 = *(const void **)v24;
    }
    v16 = v23 + 16;
    v25 = v22 + 16;
    if ( (int)sub_C4C880(v23 + 16, v22 + 16) >= 0 )
      v16 = v25;
    v41 = *(_DWORD *)(v16 + 8);
    if ( v41 > 0x40 )
      goto LABEL_52;
  }
  else
  {
    if ( v6 <= 2u )
    {
      if ( v6 )
      {
        if ( v4 <= 0x40 )
        {
          if ( *(_QWORD *)a3 != *(_QWORD *)a4 )
            goto LABEL_17;
        }
        else
        {
          v28 = a4;
          v32 = a3;
          v7 = sub_C43C50(a3, (const void **)a4);
          a3 = v32;
          a4 = v28;
          if ( !v7 )
            goto LABEL_17;
        }
        v8 = (const void **)(a3 + 16);
        if ( v5 <= 0x40 )
        {
          if ( *(_QWORD *)(a3 + 16) == *(_QWORD *)(a4 + 16) )
            goto LABEL_12;
        }
        else
        {
          v33 = a3;
          v9 = sub_C43C50(a3 + 16, (const void **)(a4 + 16));
          a3 = v33;
          if ( v9 )
          {
LABEL_12:
            *(_DWORD *)(a1 + 8) = v4;
            if ( v4 > 0x40 )
            {
              v37 = a3;
              sub_C43780(a1, (const void **)a3);
              a3 = v37;
            }
            else
            {
              *(_QWORD *)a1 = *(_QWORD *)a3;
            }
            v10 = *(_DWORD *)(a3 + 24);
            *(_DWORD *)(a1 + 24) = v10;
            if ( v10 > 0x40 )
              sub_C43780(a1 + 16, v8);
            else
              *(_QWORD *)(a1 + 16) = *(_QWORD *)(a3 + 16);
            return a1;
          }
        }
LABEL_17:
        *(_QWORD *)a1 = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 8) = 1;
        *(_DWORD *)(a1 + 24) = 1;
        return a1;
      }
      if ( v4 <= 0x40 )
      {
        v27 = *(const void **)a3;
        if ( *(_QWORD *)a3 == *(_QWORD *)a4 )
        {
          v39 = *(_DWORD *)(a3 + 8);
          v38 = v27;
          goto LABEL_42;
        }
      }
      else
      {
        v30 = a4;
        v35 = a3;
        v20 = sub_C43C50(a3, (const void **)a4);
        a3 = v35;
        a4 = v30;
        if ( v20 )
        {
          v39 = v4;
          sub_C43780((__int64)&v38, (const void **)v35);
          a3 = v35;
          a4 = v30;
          v5 = *(_DWORD *)(v35 + 24);
          goto LABEL_42;
        }
      }
      v39 = 1;
      v38 = 0;
LABEL_42:
      if ( v5 <= 0x40 )
      {
        v26 = *(const void **)(a3 + 16);
        if ( v26 == *(const void **)(a4 + 16) )
        {
          v41 = v5;
          v40 = v26;
          goto LABEL_28;
        }
        goto LABEL_44;
      }
      v16 = a3 + 16;
      if ( !sub_C43C50(a3 + 16, (const void **)(a4 + 16)) )
      {
LABEL_44:
        v41 = 1;
        v40 = 0;
        goto LABEL_28;
      }
      v41 = v5;
LABEL_52:
      sub_C43780((__int64)&v40, (const void **)v16);
      goto LABEL_28;
    }
    if ( v6 != 3 )
      BUG();
    v29 = a4;
    v34 = a3;
    v12 = sub_C4C880(a3, a4);
    v13 = v29;
    v14 = v34;
    v15 = v29;
    if ( v12 > 0 )
      v15 = v34;
    v39 = *(_DWORD *)(v15 + 8);
    if ( v39 > 0x40 )
    {
      sub_C43780((__int64)&v38, (const void **)v15);
      v13 = v29;
      v14 = v34;
    }
    else
    {
      v38 = *(const void **)v15;
    }
    v16 = v14 + 16;
    v17 = v13 + 16;
    if ( (int)sub_C4C880(v14 + 16, v13 + 16) <= 0 )
      v16 = v17;
    v41 = *(_DWORD *)(v16 + 8);
    if ( v41 > 0x40 )
      goto LABEL_52;
  }
  v40 = *(const void **)v16;
LABEL_28:
  v18 = v39;
  *(_DWORD *)(a1 + 8) = v39;
  if ( v18 > 0x40 )
    sub_C43780(a1, &v38);
  else
    *(_QWORD *)a1 = v38;
  v19 = v41;
  *(_DWORD *)(a1 + 24) = v41;
  if ( v19 > 0x40 )
  {
    sub_C43780(a1 + 16, &v40);
    if ( v41 > 0x40 && v40 )
      j_j___libc_free_0_0(v40);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = v40;
  }
  if ( v39 > 0x40 && v38 )
    j_j___libc_free_0_0(v38);
  return a1;
}
