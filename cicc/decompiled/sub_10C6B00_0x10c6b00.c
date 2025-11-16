// Function: sub_10C6B00
// Address: 0x10c6b00
//
unsigned __int8 *__fastcall sub_10C6B00(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rax
  __int64 v9; // rax
  char v10; // bl
  __int64 v11; // rdx
  bool v12; // cl
  int v13; // esi
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rcx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r12
  _QWORD *v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rcx
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // [rsp+8h] [rbp-B8h]
  __int64 v38; // [rsp+10h] [rbp-B0h]
  int v39; // [rsp+18h] [rbp-A8h]
  int v40; // [rsp+18h] [rbp-A8h]
  __int64 v41; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v42; // [rsp+28h] [rbp-98h] BYREF
  __int64 v43; // [rsp+30h] [rbp-90h] BYREF
  __int64 v44; // [rsp+38h] [rbp-88h] BYREF
  __int64 v45; // [rsp+40h] [rbp-80h] BYREF
  __int64 v46; // [rsp+48h] [rbp-78h]
  _QWORD v47[2]; // [rsp+50h] [rbp-70h] BYREF
  _BYTE v48[32]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v49; // [rsp+80h] [rbp-40h]

  v8 = *(_QWORD *)(a3 + 16);
  v41 = 0;
  v42 = 0;
  if ( v8 )
  {
    if ( !*(_QWORD *)(v8 + 8) && *(_BYTE *)a3 == 85 )
    {
      v24 = *(_QWORD *)(a3 - 32);
      if ( v24 )
      {
        if ( !*(_BYTE *)v24 && *(_QWORD *)(v24 + 24) == *(_QWORD *)(a3 + 80) && *(_DWORD *)(v24 + 36) == 207 )
        {
          v25 = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
          if ( *(_QWORD *)(a3 - 32 * v25) )
          {
            v41 = *(_QWORD *)(a3 - 32 * v25);
            v26 = *(_QWORD *)(a3 + 32 * (1 - v25));
            if ( *(_BYTE *)v26 == 17 )
            {
              if ( *(_DWORD *)(v26 + 32) <= 0x40u )
              {
                v27 = *(_QWORD *)(v26 + 24);
                goto LABEL_45;
              }
              v39 = *(_DWORD *)(v26 + 32);
              if ( v39 - (unsigned int)sub_C444A0(v26 + 24) <= 0x40 )
              {
                v27 = **(_QWORD **)(v26 + 24);
LABEL_45:
                v43 = v27;
                v28 = *(_QWORD *)(a4 + 16);
                if ( !v28 || *(_QWORD *)(v28 + 8) )
                  goto LABEL_34;
                v10 = 1;
                goto LABEL_32;
              }
            }
          }
        }
      }
    }
  }
  v9 = *(_QWORD *)(a4 + 16);
  if ( !v9 )
    goto LABEL_3;
  v10 = 0;
  if ( *(_QWORD *)(v9 + 8) )
    goto LABEL_3;
LABEL_32:
  if ( *(_BYTE *)a4 != 85 )
    goto LABEL_33;
  v33 = *(_QWORD *)(a4 - 32);
  if ( !v33 )
    goto LABEL_33;
  if ( *(_BYTE *)v33 )
    goto LABEL_33;
  if ( *(_QWORD *)(v33 + 24) != *(_QWORD *)(a4 + 80) )
    goto LABEL_33;
  if ( *(_DWORD *)(v33 + 36) != 207 )
    goto LABEL_33;
  v34 = *(_DWORD *)(a4 + 4) & 0x7FFFFFF;
  v11 = *(_QWORD *)(a4 - 32 * v34);
  if ( !v11 )
    goto LABEL_33;
  v42 = *(_QWORD *)(a4 - 32LL * (*(_DWORD *)(a4 + 4) & 0x7FFFFFF));
  v35 = *(_QWORD *)(a4 + 32 * (1 - v34));
  if ( *(_BYTE *)v35 != 17 )
    goto LABEL_33;
  if ( *(_DWORD *)(v35 + 32) <= 0x40u )
  {
    v36 = *(_QWORD *)(v35 + 24);
    goto LABEL_58;
  }
  v37 = v11;
  v38 = v35;
  v40 = *(_DWORD *)(v35 + 32);
  if ( v40 - (unsigned int)sub_C444A0(v35 + 24) > 0x40 )
  {
LABEL_33:
    if ( v10 )
    {
LABEL_34:
      v10 = 1;
LABEL_4:
      if ( sub_10B82A0(a4, &v42, &v44) )
      {
        v11 = v42;
        v12 = 0;
        goto LABEL_6;
      }
      return 0;
    }
LABEL_3:
    v10 = 0;
    if ( !sub_10B82A0(a3, &v41, &v43) )
      return 0;
    goto LABEL_4;
  }
  v11 = v37;
  v36 = **(_QWORD **)(v38 + 24);
LABEL_58:
  v44 = v36;
  if ( v10 )
  {
    v12 = v10;
  }
  else
  {
    v12 = sub_10B82A0(a3, &v41, &v43);
    if ( !v12 )
      return 0;
    v11 = v42;
  }
LABEL_6:
  if ( v41 != v11 )
    return 0;
  v13 = *a2;
  switch ( v13 )
  {
    case ':':
      v14 = (unsigned int)v43 | (unsigned int)v44;
      break;
    case ';':
      v14 = (unsigned int)v43 ^ (unsigned int)v44;
      break;
    case '9':
      v14 = (unsigned int)v43 & (unsigned int)v44;
      if ( v10 )
        goto LABEL_11;
      goto LABEL_22;
    default:
      BUG();
  }
  if ( v10 )
  {
LABEL_11:
    v15 = sub_AD64C0(*(_QWORD *)(*(_QWORD *)(a3 + 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))) + 8LL), v14, 0);
    v16 = a3 + 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
    if ( *(_QWORD *)v16 )
    {
      v17 = *(_QWORD *)(v16 + 8);
      **(_QWORD **)(v16 + 16) = v17;
      if ( v17 )
        *(_QWORD *)(v17 + 16) = *(_QWORD *)(v16 + 16);
    }
    *(_QWORD *)v16 = v15;
    if ( v15 )
    {
      v18 = *(_QWORD *)(v15 + 16);
      *(_QWORD *)(v16 + 8) = v18;
      if ( v18 )
        *(_QWORD *)(v18 + 16) = v16 + 8;
      *(_QWORD *)(v16 + 16) = v15 + 16;
      *(_QWORD *)(v15 + 16) = v16;
    }
    return sub_F162A0(a1, (__int64)a2, a3);
  }
LABEL_22:
  if ( v12 )
  {
    v20 = sub_AD64C0(*(_QWORD *)(*(_QWORD *)(a4 + 32 * (1LL - (*(_DWORD *)(a4 + 4) & 0x7FFFFFF))) + 8LL), v14, 0);
    v21 = a4 + 32 * (1LL - (*(_DWORD *)(a4 + 4) & 0x7FFFFFF));
    if ( *(_QWORD *)v21 )
    {
      v22 = *(_QWORD *)(v21 + 8);
      **(_QWORD **)(v21 + 16) = v22;
      if ( v22 )
        *(_QWORD *)(v22 + 16) = *(_QWORD *)(v21 + 16);
    }
    *(_QWORD *)v21 = v20;
    if ( v20 )
    {
      v23 = *(_QWORD *)(v20 + 16);
      *(_QWORD *)(v21 + 8) = v23;
      if ( v23 )
        *(_QWORD *)(v23 + 16) = v21 + 8;
      *(_QWORD *)(v21 + 16) = v20 + 16;
      *(_QWORD *)(v20 + 16) = v21;
    }
    return sub_F162A0(a1, (__int64)a2, a4);
  }
  else
  {
    v29 = *(_QWORD *)(a1 + 32);
    v47[0] = v41;
    v30 = *(_QWORD **)(v29 + 72);
    v49 = 257;
    HIDWORD(v46) = 0;
    v31 = sub_BCB2D0(v30);
    v47[1] = sub_ACD640(v31, v14, 0);
    v45 = *(_QWORD *)(v41 + 8);
    v32 = sub_B33D10(v29, 0xCFu, (__int64)&v45, 1, (int)v47, 2, v46, (__int64)v48);
    return sub_F162A0(a1, (__int64)a2, v32);
  }
}
