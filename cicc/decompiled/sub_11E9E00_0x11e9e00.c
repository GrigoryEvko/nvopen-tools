// Function: sub_11E9E00
// Address: 0x11e9e00
//
unsigned __int64 __fastcall sub_11E9E00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 *v7; // r12
  const char *v8; // rax
  size_t v9; // rdx
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned __int64 v12; // r15
  __int64 v14; // rdi
  __int64 *v15; // r13
  char *v16; // rax
  size_t v17; // rdx
  const char *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r8
  const char *v21; // r12
  int v22; // eax
  int v23; // eax
  int v24; // eax
  int v25; // r9d
  int v26; // eax
  int v27; // r9d
  int v28; // eax
  int v29; // r9d
  int v30; // eax
  int v31; // r9d
  int v32; // eax
  int v33; // r9d
  int v34; // eax
  int v35; // r9d
  int v36; // eax
  int v37; // r9d
  int v38; // eax
  int v39; // r9d
  int v40; // eax
  int v41; // r9d
  int v42; // eax
  int v43; // r9d
  __int64 v44; // [rsp+8h] [rbp-48h]
  unsigned int v45[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v5 = sub_B43CA0(a2);
  v6 = *(_QWORD *)(a2 - 32);
  v7 = (__int64 *)v5;
  if ( v6 )
  {
    if ( *(_BYTE *)v6 )
    {
      v6 = 0;
    }
    else if ( *(_QWORD *)(a2 + 80) != *(_QWORD *)(v6 + 24) )
    {
      v6 = 0;
    }
  }
  v8 = sub_BD5D20(v6);
  if ( !*(_BYTE *)(a1 + 80) )
    goto LABEL_9;
  if ( v9 == 3 )
  {
    if ( *(_WORD *)v8 != 24948 || v8[2] != 110 )
      goto LABEL_9;
  }
  else if ( v9 == 5 )
  {
    if ( (*(_DWORD *)v8 != 1851880545 || v8[4] != 104) && (*(_DWORD *)v8 != 1852404577 || v8[4] != 104) )
      goto LABEL_9;
  }
  else if ( v9 != 4 || *(_DWORD *)v8 != 1752066419 && *(_DWORD *)v8 != 1752395619 )
  {
    goto LABEL_9;
  }
  if ( (unsigned __int8)sub_11E9B60(a1, v7, (__int64)v8, v9, v10, v11) )
  {
    v12 = sub_11DB650(a2, a3, 0, *(__int64 **)(a1 + 24), 1);
    goto LABEL_10;
  }
LABEL_9:
  v12 = 0;
LABEL_10:
  if ( **(_BYTE **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) != 85 )
    return v12;
  v44 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  if ( !sub_B45190(a2) )
    return v12;
  if ( !sub_B45190(v44) )
    return v12;
  v14 = *(_QWORD *)(v44 - 32);
  if ( !v14 )
    return v12;
  if ( *(_BYTE *)v14 )
    return v12;
  if ( *(_QWORD *)(v14 + 24) != *(_QWORD *)(v44 + 80) )
    return v12;
  v15 = *(__int64 **)(a1 + 24);
  v16 = (char *)sub_BD5D20(v14);
  if ( !(unsigned __int8)sub_980AF0(*v15, v16, v17, v45) || !sub_11C99B0(v7, *(__int64 **)(a1 + 24), v45[0]) )
    return v12;
  v18 = sub_BD5D20(v6);
  v20 = v44;
  v21 = v18;
  switch ( v19 )
  {
    case 3LL:
      v22 = memcmp(v18, "tan", 3u);
      v20 = v44;
      if ( v22 )
      {
LABEL_32:
        v23 = 523;
        break;
      }
      v23 = 173;
      break;
    case 5LL:
      v24 = memcmp(v18, "atanh", 5u);
      v20 = v44;
      v25 = v24;
      v23 = 492;
      if ( v25 )
      {
        v26 = memcmp(v21, "sinhf", 5u);
        v20 = v44;
        v27 = v26;
        v23 = 170;
        if ( v27 )
        {
          v28 = memcmp(v21, "coshf", 5u);
          v20 = v44;
          v29 = v28;
          v23 = 163;
          if ( v29 )
          {
            v30 = memcmp(v21, "sinhl", 5u);
            v20 = v44;
            v31 = v30;
            v23 = 171;
            if ( v31 )
            {
              v32 = memcmp(v21, "coshl", 5u);
              v20 = v44;
              v33 = v32;
              v23 = 164;
              if ( v33 )
              {
                v34 = memcmp(v21, "asinh", 5u);
                v20 = v44;
                v35 = v34;
                v23 = 438;
                if ( v35 )
                  goto LABEL_32;
              }
            }
          }
        }
      }
      break;
    case 4LL:
      v23 = 169;
      if ( *(_DWORD *)v21 != 1752066419 )
      {
        v23 = 162;
        if ( *(_DWORD *)v21 != 1752395619 )
        {
          v23 = 177;
          if ( *(_DWORD *)v21 != 1718509940 )
          {
            v23 = 181;
            if ( *(_DWORD *)v21 != 1819173236 )
              goto LABEL_32;
          }
        }
      }
      break;
    case 6LL:
      v36 = memcmp(v18, "atanhf", 6u);
      v20 = v44;
      v37 = v36;
      v23 = 493;
      if ( v37 )
      {
        v38 = memcmp(v21, "atanhl", 6u);
        v20 = v44;
        v39 = v38;
        v23 = 494;
        if ( v39 )
        {
          v40 = memcmp(v21, "asinhf", 6u);
          v20 = v44;
          v41 = v40;
          v23 = 439;
          if ( v41 )
          {
            v42 = memcmp(v21, "asinhl", 6u);
            v20 = v44;
            v43 = v42;
            v23 = 440;
            if ( v43 )
              goto LABEL_32;
          }
        }
      }
      break;
    default:
      goto LABEL_32;
  }
  if ( v45[0] == v23 )
    return *(_QWORD *)(v20 - 32LL * (*(_DWORD *)(v20 + 4) & 0x7FFFFFF));
  return v12;
}
