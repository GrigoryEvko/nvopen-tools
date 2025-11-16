// Function: sub_2B6E610
// Address: 0x2b6e610
//
__int64 __fastcall sub_2B6E610(__int64 a1)
{
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // rax
  _BYTE *v9; // rdx
  unsigned int v10; // r15d
  __int64 v12; // r12
  __int64 v13; // rcx
  int v14; // eax
  _BYTE *v15; // rax
  __int64 *v16; // rdi
  __int64 v17; // r8
  unsigned __int8 **v18; // rbx
  unsigned __int8 **v19; // rax
  unsigned __int8 *v20; // r8
  unsigned __int8 *v21; // rcx
  __int64 v22; // r9
  __int64 v23; // rdx
  __int64 *v24; // rax
  unsigned __int8 v25; // dl
  __int64 v26; // rdx
  __int64 *v27; // r10
  unsigned __int8 v28; // dl
  unsigned __int8 v29; // dl
  unsigned __int8 v30; // dl
  unsigned __int8 v31; // dl
  unsigned __int8 v32; // dl
  unsigned __int8 v33; // dl
  __int64 v34; // [rsp+8h] [rbp-38h]

  v2 = *(unsigned int *)(a1 + 3556);
  v3 = *(unsigned int *)(a1 + 8);
  if ( (_DWORD)v2 != (_DWORD)v3 )
  {
    if ( !*(_DWORD *)(**(_QWORD **)a1 + 8LL) || (v4 = *(unsigned int *)(**(_QWORD **)a1 + 8LL), (v4 & (v4 - 1)) != 0) )
    {
      if ( (unsigned int)v2 <= 3 )
      {
        v5 = *(_QWORD *)a1 + 8 * v2;
        v6 = v5 + 8 * (v3 - v2);
        if ( v5 != v6 )
        {
          v7 = 0;
          do
          {
            v8 = *(_QWORD *)v5;
            if ( *(_DWORD *)(*(_QWORD *)v5 + 104LL) == 3 )
            {
              v9 = *(_BYTE **)(v8 + 416);
              if ( v9 )
              {
                if ( *(_QWORD *)(v8 + 424) && *v9 == 61 )
                  v7 += (unsigned __int8)sub_2B17600(*(_BYTE ***)v8, *(unsigned int *)(v8 + 8)) == 0;
              }
            }
            v5 += 8;
          }
          while ( v6 != v5 );
          v10 = 1;
          if ( v7 == 1 )
            return v10;
        }
      }
    }
    return 0;
  }
  v12 = 0;
  v10 = 0;
  if ( !*(_DWORD *)(a1 + 3556) )
    return 0;
  do
  {
    v13 = *(_QWORD *)(*(_QWORD *)a1 + 8LL * (unsigned int)v12);
    v14 = *(_DWORD *)(v13 + 104);
    if ( v14 == 5 )
      return 0;
    if ( v14 != 3 )
      goto LABEL_30;
    v15 = *(_BYTE **)(v13 + 416);
    if ( v15 && *(_QWORD *)(v13 + 424) )
    {
      if ( *v15 != 61 )
        return 0;
      v16 = *(__int64 **)v13;
      v17 = *(unsigned int *)(v13 + 8);
      if ( ***(_BYTE ***)v13 == 90 )
        goto LABEL_40;
      goto LABEL_22;
    }
    v17 = *(unsigned int *)(v13 + 8);
    v16 = *(__int64 **)v13;
    v22 = *(_QWORD *)v13 + 8 * v17;
    v23 = (8 * v17) >> 3;
    if ( (8 * v17) >> 5 )
    {
      v24 = *(__int64 **)v13;
      while ( 1 )
      {
        v25 = *(_BYTE *)*v24;
        if ( v25 <= 0x1Cu || v25 != 61 && v25 != 90 )
          goto LABEL_38;
        v27 = v24 + 1;
        v28 = *(_BYTE *)v24[1];
        if ( v28 <= 0x1Cu || v28 != 90 && v28 != 61 )
          break;
        v27 = v24 + 2;
        v29 = *(_BYTE *)v24[2];
        if ( v29 <= 0x1Cu || v29 != 61 && v29 != 90 )
          break;
        v27 = v24 + 3;
        v30 = *(_BYTE *)v24[3];
        if ( v30 <= 0x1Cu || v30 != 61 && v30 != 90 )
          break;
        v24 += 4;
        if ( &v16[4 * ((8 * v17) >> 5)] == v24 )
        {
          v23 = (v22 - (__int64)v24) >> 3;
          goto LABEL_57;
        }
      }
      v24 = v27;
      goto LABEL_38;
    }
    v24 = *(__int64 **)v13;
LABEL_57:
    if ( v23 == 2 )
      goto LABEL_63;
    if ( v23 == 3 )
    {
      v31 = *(_BYTE *)*v24;
      if ( v31 <= 0x1Cu || v31 != 61 && v31 != 90 )
        goto LABEL_38;
      ++v24;
LABEL_63:
      v32 = *(_BYTE *)*v24;
      if ( v32 <= 0x1Cu || v32 != 61 && v32 != 90 )
        goto LABEL_38;
      ++v24;
      goto LABEL_70;
    }
    if ( v23 != 1 )
      return 0;
LABEL_70:
    v33 = *(_BYTE *)*v24;
    if ( v33 > 0x1Cu && (v33 == 90 || v33 == 61) )
      return 0;
LABEL_38:
    if ( (__int64 *)v22 == v24 )
      return 0;
    if ( *(_BYTE *)*v16 == 90 )
    {
LABEL_40:
      v34 = *(_QWORD *)(*(_QWORD *)a1 + 8LL * (unsigned int)v12);
      if ( sub_2B5F980(v16 + 1, v17 - 1, *(__int64 **)(a1 + 3304)) && v26 )
        return 0;
      v16 = *(__int64 **)v34;
      v17 = *(unsigned int *)(v34 + 8);
    }
LABEL_22:
    v18 = (unsigned __int8 **)&v16[v17];
    if ( v18 == (unsigned __int8 **)v16 )
    {
LABEL_28:
      if ( v18 != sub_2B0BF30(v16, (__int64)v18, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B0D8B0) )
        v10 = 1;
      goto LABEL_30;
    }
    v19 = (unsigned __int8 **)v16;
    v20 = 0;
    do
    {
      while ( 1 )
      {
        v21 = *v19;
        if ( (unsigned int)**v19 - 12 > 1 )
          break;
LABEL_24:
        if ( v18 == ++v19 )
          goto LABEL_33;
      }
      if ( v20 )
      {
        if ( v21 != v20 )
          goto LABEL_28;
        goto LABEL_24;
      }
      ++v19;
      v20 = v21;
    }
    while ( v18 != v19 );
LABEL_33:
    if ( !v20 )
      goto LABEL_28;
LABEL_30:
    ++v12;
  }
  while ( v2 != v12 );
  return v10;
}
