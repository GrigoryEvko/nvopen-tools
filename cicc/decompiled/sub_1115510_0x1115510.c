// Function: sub_1115510
// Address: 0x1115510
//
_QWORD *__fastcall sub_1115510(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // r13
  int v8; // r14d
  __int64 v9; // r13
  unsigned int v10; // edx
  int v11; // eax
  bool v12; // al
  __int64 v13; // rax
  __int64 v14; // r13
  _QWORD *v15; // r12
  bool v16; // al
  __int64 v18; // rdx
  _BYTE *v19; // rax
  bool v20; // al
  bool v21; // al
  __int64 v22; // rax
  bool v23; // al
  int v24; // eax
  int v25; // ecx
  __int64 v26; // rdx
  __int64 v27; // rcx
  unsigned int v28; // r8d
  unsigned int v29; // edx
  unsigned int v30; // [rsp+Ch] [rbp-94h]
  int v31; // [rsp+Ch] [rbp-94h]
  __int16 v32; // [rsp+10h] [rbp-90h]
  __int64 v33; // [rsp+10h] [rbp-90h]
  int v34; // [rsp+18h] [rbp-88h]
  unsigned int v35; // [rsp+18h] [rbp-88h]
  bool v36; // [rsp+18h] [rbp-88h]
  unsigned int v37; // [rsp+18h] [rbp-88h]
  unsigned int v38; // [rsp+18h] [rbp-88h]
  __int64 v39; // [rsp+20h] [rbp-80h]
  __int64 v40; // [rsp+28h] [rbp-78h]
  __int64 v41; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v42; // [rsp+38h] [rbp-68h]
  __int64 v43; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v44; // [rsp+48h] [rbp-58h]
  __int16 v45; // [rsp+60h] [rbp-40h]

  v6 = a2;
  v7 = *(_QWORD *)(a3 - 32);
  v8 = *(_WORD *)(a2 + 2) & 0x3F;
  v32 = *(_WORD *)(a2 + 2) & 0x3F;
  v39 = *(_QWORD *)(a3 + 8);
  v40 = *(_QWORD *)(a3 - 64);
  if ( (unsigned int)(v8 - 32) <= 1 )
  {
    if ( *(_DWORD *)(a4 + 8) <= 0x40u )
    {
      v20 = *(_QWORD *)a4 == 0;
    }
    else
    {
      v31 = *(_DWORD *)(a4 + 8);
      v20 = v31 == (unsigned int)sub_C444A0(a4);
    }
    if ( v20 && v40 == v7 && (sub_B448F0(a3) || sub_B44900(a3)) )
      goto LABEL_14;
  }
  if ( *(_BYTE *)v7 == 17 )
  {
    v9 = v7 + 24;
  }
  else
  {
    v18 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17;
    if ( (unsigned int)v18 > 1 )
      return 0;
    if ( *(_BYTE *)v7 > 0x15u )
      return 0;
    a2 = 0;
    v19 = sub_AD7630(v7, 0, v18);
    if ( !v19 || *v19 != 17 )
      return 0;
    v9 = (__int64)(v19 + 24);
  }
  if ( !sub_B532B0(v8) )
    goto LABEL_22;
  v10 = *(_DWORD *)(a4 + 8);
  if ( v10 <= 0x40 )
  {
    v12 = *(_QWORD *)a4 == 0;
  }
  else
  {
    v30 = *(_DWORD *)(a4 + 8);
    v11 = sub_C444A0(a4);
    v10 = v30;
    v12 = v30 == v11;
  }
  if ( !v12 )
  {
    if ( v10 > 0x40 )
    {
      v37 = v10;
      v24 = sub_C444A0(a4);
      v10 = v37;
      if ( v24 != v37 - 1 )
      {
LABEL_18:
        if ( !v10
          || (v10 <= 0x40
            ? (v16 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v10) == *(_QWORD *)a4)
            : (v16 = v10 == (unsigned int)sub_C445E0(a4)),
              v16) )
        {
          if ( v32 == 38 )
          {
            v8 = 39;
            goto LABEL_9;
          }
        }
LABEL_22:
        if ( *(_DWORD *)(v9 + 8) <= 0x40u )
        {
          if ( !*(_QWORD *)v9 )
            return 0;
          if ( (*(_WORD *)(v6 + 2) & 0x3Fu) - 32 > 1 )
            goto LABEL_25;
        }
        else
        {
          v34 = *(_DWORD *)(v9 + 8);
          if ( v34 == (unsigned int)sub_C444A0(v9) )
            return 0;
          if ( (*(_WORD *)(v6 + 2) & 0x3Fu) - 32 > 1 )
          {
LABEL_25:
            if ( sub_B44900(a3) && sub_B532B0(v8) )
            {
              if ( (unsigned __int8)sub_986B30((__int64 *)a4, a2, v26, v27, v28) && sub_986760(v9) )
                return 0;
              if ( sub_986C60((__int64 *)v9, *(_DWORD *)(v9 + 8) - 1) )
                v8 = sub_B52F50(v8);
              if ( (unsigned int)(v8 - 39) <= 1 )
                sub_C4CAA0((__int64)&v43, a4, v9, 2);
              else
                sub_C4CAA0((__int64)&v43, a4, v9, 0);
            }
            else
            {
              if ( !sub_B448F0(a3) || !sub_B532A0(v8) )
                return 0;
              v25 = 2;
              if ( (unsigned int)(v8 - 35) > 1 )
                v25 = 0;
              sub_C4C950((__int64)&v43, a4, v9, v25);
            }
            v14 = sub_AD8D80(v39, (__int64)&v43);
            sub_969240(&v43);
            if ( v14 )
              goto LABEL_62;
            return 0;
          }
        }
        if ( !sub_B44900(a3) )
        {
LABEL_44:
          a2 = a4;
          sub_C4B490((__int64)&v43, a4, v9);
          if ( v44 <= 0x40 )
          {
            v21 = v43 == 0;
          }
          else
          {
            v35 = v44;
            v21 = v35 == (unsigned int)sub_C444A0((__int64)&v43);
            if ( v43 )
            {
              v36 = v21;
              j_j___libc_free_0_0(v43);
              v21 = v36;
            }
          }
          if ( !v21 )
            goto LABEL_25;
          a2 = v9;
          sub_9865C0((__int64)&v41, v9);
          if ( v42 > 0x40 )
          {
            *(_QWORD *)v41 &= 1uLL;
            a2 = 0;
            v33 = v41;
            memset((void *)(v41 + 8), 0, 8 * (unsigned int)(((unsigned __int64)v42 + 63) >> 6) - 8);
            v29 = v42;
            v42 = 0;
            v44 = v29;
            v22 = v33;
            v43 = v33;
            if ( v29 > 0x40 )
            {
              v23 = v29 - 1 == (unsigned int)sub_C444A0((__int64)&v43);
LABEL_51:
              if ( !v23 && !sub_B448F0(a3) )
              {
                sub_969240(&v43);
                sub_969240(&v41);
                goto LABEL_25;
              }
              sub_969240(&v43);
              sub_969240(&v41);
              sub_C4A1D0((__int64)&v43, a4, v9);
              v14 = sub_AD8D80(v39, (__int64)&v43);
              sub_969240(&v43);
LABEL_62:
              v45 = 257;
              v15 = sub_BD2C40(72, unk_3F10FD0);
              if ( !v15 )
                return v15;
              goto LABEL_15;
            }
          }
          else
          {
            v22 = v41 & 1;
            v44 = v42;
            v41 = v22;
            v43 = (unsigned int)v22;
            v42 = 0;
          }
          v23 = v22 == 1;
          goto LABEL_51;
        }
        sub_C4B8A0((__int64)&v43, a4, v9);
        if ( v44 <= 0x40 )
        {
          if ( v43 )
            goto LABEL_44;
        }
        else
        {
          v38 = v44;
          if ( v38 != (unsigned int)sub_C444A0((__int64)&v43) )
          {
            if ( v43 )
              j_j___libc_free_0_0(v43);
            goto LABEL_44;
          }
          if ( v43 )
            j_j___libc_free_0_0(v43);
        }
        sub_C4A3E0((__int64)&v43, a4, v9);
        v14 = sub_AD8D80(v39, (__int64)&v43);
        if ( v44 > 0x40 && v43 )
          j_j___libc_free_0_0(v43);
        goto LABEL_62;
      }
    }
    else if ( *(_QWORD *)a4 != 1 )
    {
      goto LABEL_18;
    }
    if ( v32 == 40 )
    {
      v8 = 41;
      goto LABEL_9;
    }
    goto LABEL_22;
  }
  if ( (unsigned int)(v8 - 32) <= 1 )
    goto LABEL_22;
LABEL_9:
  if ( !sub_B44900(a3) )
    goto LABEL_22;
  a2 = *(unsigned int *)(v9 + 8);
  v13 = *(_QWORD *)v9;
  if ( (unsigned int)a2 > 0x40 )
    v13 = *(_QWORD *)(v13 + 8LL * ((unsigned int)(a2 - 1) >> 6));
  if ( (v13 & (1LL << ((unsigned __int8)a2 - 1))) != 0 )
    LOWORD(v8) = sub_B52F50(v8);
LABEL_14:
  v14 = sub_AD6530(v39, a2);
  v45 = 257;
  v15 = sub_BD2C40(72, unk_3F10FD0);
  if ( v15 )
LABEL_15:
    sub_1113300((__int64)v15, v8, v40, v14, (__int64)&v43);
  return v15;
}
