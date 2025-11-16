// Function: sub_104A1B0
// Address: 0x104a1b0
//
__int64 __fastcall sub_104A1B0(_BYTE *a1, __int64 a2, bool a3, _QWORD *a4)
{
  bool v7; // r15
  bool v8; // al
  __int64 v9; // r9
  int v10; // eax
  bool v12; // zf
  __int64 v13; // r13
  unsigned int v14; // r14d
  __int64 v15; // rsi
  __int64 v16; // rax
  _DWORD *v17; // rax
  __int64 v18; // rax
  int v19; // eax
  unsigned int v20; // edx
  bool v21; // al
  __int64 v22; // r14
  __int64 v23; // rdx
  _BYTE *v24; // rax
  unsigned int v25; // r14d
  _BYTE *v27; // rax
  unsigned __int8 *v28; // r9
  bool v29; // r8
  unsigned int v30; // edx
  __int64 v31; // rsi
  __int64 v32; // rax
  int v33; // eax
  unsigned int v34; // r14d
  __int64 v35; // rax
  unsigned int v36; // r15d
  int v37; // [rsp-ACh] [rbp-ACh]
  unsigned __int8 *v38; // [rsp-A8h] [rbp-A8h]
  bool v39; // [rsp-A8h] [rbp-A8h]
  int v40; // [rsp-A8h] [rbp-A8h]
  __int64 v41; // [rsp-98h] [rbp-98h]
  __int64 v42; // [rsp-98h] [rbp-98h]
  unsigned __int8 *v43; // [rsp-98h] [rbp-98h]
  __int64 v44; // [rsp-90h] [rbp-90h]
  int v45; // [rsp-90h] [rbp-90h]
  _DWORD v46[2]; // [rsp-80h] [rbp-80h] BYREF
  __int64 v47; // [rsp-78h] [rbp-78h] BYREF
  __int64 v48; // [rsp-70h] [rbp-70h] BYREF
  _QWORD *v49; // [rsp-68h] [rbp-68h] BYREF
  __int64 *v50; // [rsp-60h] [rbp-60h]
  _QWORD v51[11]; // [rsp-58h] [rbp-58h] BYREF

  if ( *a1 != 82 )
    return 0;
  v44 = *((_QWORD *)a1 - 8);
  if ( !v44 || **((_BYTE **)a1 - 4) > 0x15u )
    return 0;
  v41 = *((_QWORD *)a1 - 4);
  v7 = a3;
  v8 = sub_AC30F0(v41);
  v9 = v41;
  if ( !v8 )
  {
    if ( *(_BYTE *)v41 == 17 )
    {
      v20 = *(_DWORD *)(v41 + 32);
      if ( v20 <= 0x40 )
        v21 = *(_QWORD *)(v41 + 24) == 0;
      else
        v21 = v20 == (unsigned int)sub_C444A0(v41 + 24);
    }
    else
    {
      v42 = *(_QWORD *)(v41 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v42 + 8) - 17 > 1 )
        return 0;
      v38 = (unsigned __int8 *)v9;
      v27 = sub_AD7630(v9, 0, v42);
      v28 = v38;
      v29 = 0;
      if ( !v27 || *v27 != 17 )
      {
        if ( *(_BYTE *)(v42 + 8) == 17 )
        {
          v37 = *(_DWORD *)(v42 + 32);
          if ( v37 )
          {
            v31 = 0;
            while ( 1 )
            {
              v39 = v29;
              v43 = v28;
              v32 = sub_AD69F0(v28, v31);
              if ( !v32 )
                break;
              v28 = v43;
              v29 = v39;
              if ( *(_BYTE *)v32 != 13 )
              {
                if ( *(_BYTE *)v32 != 17 )
                  break;
                if ( *(_DWORD *)(v32 + 32) <= 0x40u )
                {
                  v29 = *(_QWORD *)(v32 + 24) == 0;
                }
                else
                {
                  v40 = *(_DWORD *)(v32 + 32);
                  v33 = sub_C444A0(v32 + 24);
                  v28 = v43;
                  v29 = v40 == v33;
                }
                if ( !v29 )
                  break;
              }
              v31 = (unsigned int)(v31 + 1);
              if ( v37 == (_DWORD)v31 )
              {
                if ( v29 )
                  goto LABEL_5;
                return 0;
              }
            }
          }
        }
        return 0;
      }
      v30 = *((_DWORD *)v27 + 8);
      if ( v30 <= 0x40 )
        v21 = *((_QWORD *)v27 + 3) == 0;
      else
        v21 = v30 == (unsigned int)sub_C444A0((__int64)(v27 + 24));
    }
    if ( v21 )
      goto LABEL_5;
    return 0;
  }
LABEL_5:
  v10 = sub_B53900((__int64)a1);
  v51[1] = &v48;
  v51[0] = v44;
  v51[2] = v46;
  if ( a3 )
  {
    if ( v10 != 33 )
      return 0;
    if ( *(_BYTE *)a2 != 93 )
      return 0;
    v19 = *(_DWORD *)(a2 + 80);
    LODWORD(v49) = 1;
    if ( v19 != 1 || **(_DWORD **)(a2 + 72) != 1 || !(unsigned __int8)sub_104A0A0((__int64)v51, a2) )
      return 0;
    goto LABEL_24;
  }
  if ( v10 != 32 )
    return 0;
  v12 = *(_BYTE *)a2 == 59;
  v49 = 0;
  v50 = &v47;
  if ( !v12 )
    return 0;
  v13 = *(_QWORD *)(a2 - 64);
  if ( *(_BYTE *)v13 == 17 )
  {
    v14 = *(_DWORD *)(v13 + 32);
    if ( v14 )
    {
      if ( v14 <= 0x40 )
      {
        if ( *(_QWORD *)(v13 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v14) )
          goto LABEL_16;
      }
      else if ( v14 != (unsigned int)sub_C445E0(v13 + 24) )
      {
        goto LABEL_16;
      }
    }
  }
  else
  {
    v22 = *(_QWORD *)(v13 + 8);
    v23 = (unsigned int)*(unsigned __int8 *)(v22 + 8) - 17;
    if ( (unsigned int)v23 > 1 || *(_BYTE *)v13 > 0x15u )
      goto LABEL_16;
    v24 = sub_AD7630(*(_QWORD *)(a2 - 64), 0, v23);
    if ( !v24 || *v24 != 17 )
    {
      if ( *(_BYTE *)(v22 + 8) == 17 )
      {
        v45 = *(_DWORD *)(v22 + 32);
        if ( v45 )
        {
          v34 = 0;
          while ( 1 )
          {
            v35 = sub_AD69F0((unsigned __int8 *)v13, v34);
            if ( !v35 )
              break;
            if ( *(_BYTE *)v35 != 13 )
            {
              if ( *(_BYTE *)v35 != 17 )
                goto LABEL_16;
              v36 = *(_DWORD *)(v35 + 32);
              if ( v36 )
              {
                if ( v36 <= 0x40 )
                  v7 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v36) == *(_QWORD *)(v35 + 24);
                else
                  v7 = v36 == (unsigned int)sub_C445E0(v35 + 24);
                if ( !v7 )
                  goto LABEL_16;
              }
              else
              {
                v7 = 1;
              }
            }
            if ( v45 == ++v34 )
            {
              if ( v7 )
                goto LABEL_45;
              goto LABEL_16;
            }
          }
        }
      }
      goto LABEL_16;
    }
    v25 = *((_DWORD *)v24 + 8);
    if ( v25 )
    {
      if ( !(v25 <= 0x40
           ? 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v25) == *((_QWORD *)v24 + 3)
           : v25 == (unsigned int)sub_C445E0((__int64)(v24 + 24))) )
      {
LABEL_16:
        v15 = *(_QWORD *)(a2 - 32);
        goto LABEL_17;
      }
    }
LABEL_45:
    if ( v49 )
      *v49 = v13;
  }
  v15 = *(_QWORD *)(a2 - 32);
  if ( v15 )
  {
    *v50 = v15;
    goto LABEL_20;
  }
LABEL_17:
  if ( !(unsigned __int8)sub_995B10(&v49, v15) )
    return 0;
  v16 = *(_QWORD *)(a2 - 64);
  if ( !v16 )
    return 0;
  *v50 = v16;
LABEL_20:
  if ( *(_BYTE *)v47 != 93 )
    return 0;
  v12 = *(_DWORD *)(v47 + 80) == 1;
  v17 = *(_DWORD **)(v47 + 72);
  v46[1] = 1;
  if ( !v12 || *v17 != 1 || !(unsigned __int8)sub_104A0A0((__int64)v51, v47) )
    return 0;
LABEL_24:
  if ( (*(_BYTE *)(v48 + 7) & 0x40) != 0 )
    v18 = *(_QWORD *)(v48 - 8);
  else
    v18 = v48 - 32LL * (*(_DWORD *)(v48 + 4) & 0x7FFFFFF);
  *a4 = 32LL * (v46[0] == 0) + v18;
  return 1;
}
