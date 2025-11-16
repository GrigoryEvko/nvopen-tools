// Function: sub_2D04C10
// Address: 0x2d04c10
//
unsigned __int64 __fastcall sub_2D04C10(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 result; // rax
  unsigned int v8; // edx
  __int64 v9; // r9
  __int64 *v10; // r13
  __int64 v11; // r8
  _BYTE **v12; // rcx
  __int64 v13; // r8
  _BYTE *v14; // rsi
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  __int64 v17; // rax
  _QWORD *v18; // rax
  _QWORD *v19; // rdx
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  __int64 *v22; // rax
  __int64 *v23; // rax
  __int64 *v24; // rax
  _BYTE **v25; // [rsp+8h] [rbp-48h]
  unsigned __int8 v26; // [rsp+10h] [rbp-40h]
  __int64 *v27; // [rsp+10h] [rbp-40h]
  __int64 *v28; // [rsp+10h] [rbp-40h]
  __int64 v29; // [rsp+18h] [rbp-38h]
  __int64 v30; // [rsp+18h] [rbp-38h]
  __int64 v31; // [rsp+18h] [rbp-38h]
  __int64 *v32; // [rsp+18h] [rbp-38h]

  result = (unsigned __int64)sub_AE6EC0(a4, a1);
  if ( !(_BYTE)v8 )
    return result;
  v9 = v8;
  v10 = (__int64 *)a1;
  v11 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v12 = *(_BYTE ***)(a1 - 8);
    v13 = (__int64)&v12[v11];
  }
  else
  {
    v12 = (_BYTE **)(a1 - v11 * 8);
    v13 = a1;
  }
  if ( (_BYTE **)v13 == v12 )
    goto LABEL_29;
  do
  {
    v14 = *v12;
    if ( **v12 <= 0x1Cu )
      goto LABEL_12;
    if ( *(_BYTE *)(a2 + 28) )
    {
      v15 = *(_QWORD **)(a2 + 8);
      v16 = &v15[*(unsigned int *)(a2 + 20)];
      if ( v15 != v16 )
      {
        while ( v14 != (_BYTE *)*v15 )
        {
          if ( v16 == ++v15 )
            goto LABEL_12;
        }
LABEL_11:
        v9 = 0;
      }
    }
    else
    {
      v25 = v12;
      v26 = v9;
      v29 = v13;
      v22 = sub_C8CA60(a2, (__int64)v14);
      v13 = v29;
      v9 = v26;
      v12 = v25;
      if ( v22 )
        goto LABEL_11;
    }
LABEL_12:
    v12 += 4;
  }
  while ( (_BYTE **)v13 != v12 );
  if ( !(_BYTE)v9 )
  {
    v17 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
    v9 = a1 - v17;
    if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    {
      v9 = *(_QWORD *)(a1 - 8);
      v10 = (__int64 *)(v9 + v17);
    }
    for ( ; v10 != (__int64 *)v9; v9 += 32 )
    {
      v13 = *(_QWORD *)v9;
      if ( **(_BYTE **)v9 > 0x1Cu )
      {
        if ( *(_BYTE *)(a2 + 28) )
        {
          v18 = *(_QWORD **)(a2 + 8);
          v19 = &v18[*(unsigned int *)(a2 + 20)];
          if ( v18 == v19 )
            continue;
          while ( v13 != *v18 )
          {
            if ( v19 == ++v18 )
              goto LABEL_28;
          }
        }
        else
        {
          v27 = (__int64 *)v9;
          v30 = *(_QWORD *)v9;
          v23 = sub_C8CA60(a2, *(_QWORD *)v9);
          v13 = v30;
          v9 = (__int64)v27;
          if ( !v23 )
            continue;
        }
        if ( *(_BYTE *)(a4 + 28) )
        {
          v20 = *(_QWORD **)(a4 + 8);
          v21 = &v20[*(unsigned int *)(a4 + 20)];
          if ( v20 != v21 )
          {
            while ( v13 != *v20 )
            {
              if ( v21 == ++v20 )
                goto LABEL_39;
            }
            continue;
          }
LABEL_39:
          v32 = (__int64 *)v9;
          sub_2D04C10(v13, a2, a3, a4);
          v9 = (__int64)v32;
          continue;
        }
        v28 = (__int64 *)v9;
        v31 = v13;
        v24 = sub_C8CA60(a4, v13);
        v13 = v31;
        v9 = (__int64)v28;
        if ( !v24 )
          goto LABEL_39;
      }
LABEL_28:
      ;
    }
  }
LABEL_29:
  result = *(unsigned int *)(a3 + 8);
  if ( result + 1 > *(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), result + 1, 8u, v13, v9);
    result = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * result) = a1;
  ++*(_DWORD *)(a3 + 8);
  return result;
}
