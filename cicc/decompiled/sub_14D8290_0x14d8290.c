// Function: sub_14D8290
// Address: 0x14d8290
//
__int64 __fastcall sub_14D8290(__int64 a1, __int64 a2, _BYTE *a3)
{
  char v6; // al
  __int64 v7; // r14
  __int16 v9; // ax
  __int64 v10; // rbx
  unsigned int v11; // eax
  char v12; // dl
  unsigned __int8 *v13; // rbx
  unsigned int v14; // edx
  unsigned int v15; // esi
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rbx
  _BYTE *v21; // rbx
  __int64 v22; // rdi
  __int64 *v23; // rdi
  __int64 v24; // rax
  unsigned __int8 *v25; // r12
  unsigned int v26; // edx
  unsigned int v27; // esi
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rax
  __int64 v30; // rax
  unsigned int v31; // ebx
  unsigned int v32; // edx
  unsigned __int64 v33; // r12
  unsigned __int64 v34; // r12
  unsigned int v35; // [rsp+Ch] [rbp-84h]
  unsigned __int64 v36; // [rsp+10h] [rbp-80h]
  unsigned __int64 v37; // [rsp+10h] [rbp-80h]
  unsigned __int8 *v38; // [rsp+18h] [rbp-78h]
  unsigned int v39; // [rsp+18h] [rbp-78h]
  unsigned int v40; // [rsp+18h] [rbp-78h]
  unsigned __int8 *v41; // [rsp+20h] [rbp-70h] BYREF
  __int64 v42; // [rsp+28h] [rbp-68h]
  unsigned __int64 v43; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v44; // [rsp+38h] [rbp-58h]
  unsigned __int64 v45; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v46; // [rsp+48h] [rbp-48h]
  unsigned __int64 v47; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v48; // [rsp+58h] [rbp-38h]

  while ( 2 )
  {
    v6 = *(_BYTE *)(a1 + 16);
    if ( v6 == 3 )
    {
      if ( (*(_BYTE *)(a1 + 80) & 1) == 0 )
        return 0;
      if ( !(unsigned __int8)sub_15E4F60(a1) )
      {
        switch ( *(_BYTE *)(a1 + 32) & 0xF )
        {
          case 0:
          case 1:
          case 3:
          case 5:
          case 6:
          case 7:
          case 8:
            if ( (*(_BYTE *)(a1 + 80) & 2) != 0 )
              goto LABEL_6;
            v7 = *(_QWORD *)(a1 - 24);
            break;
          case 2:
          case 4:
          case 9:
          case 0xA:
            goto LABEL_6;
          default:
LABEL_119:
            JUMPOUT(0x419798);
        }
        return v7;
      }
LABEL_6:
      v6 = *(_BYTE *)(a1 + 16);
    }
    if ( v6 == 1 )
    {
      v7 = *(_QWORD *)(a1 - 24);
      if ( v7 )
      {
        switch ( *(_BYTE *)(a1 + 32) & 0xF )
        {
          case 0:
          case 1:
          case 3:
          case 5:
          case 6:
          case 7:
          case 8:
            a1 = *(_QWORD *)(a1 - 24);
            continue;
          case 2:
          case 4:
          case 9:
          case 0xA:
            return 0;
          default:
            goto LABEL_119;
        }
      }
      return v7;
    }
    break;
  }
  if ( v6 != 5 )
    return 0;
  v9 = *(_WORD *)(a1 + 18);
  if ( v9 != 32 )
    goto LABEL_13;
  v21 = *(_BYTE **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  if ( v21[16] == 3 && (v21[80] & 1) != 0 )
  {
    if ( !(unsigned __int8)sub_15E4F60(*(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF))) )
    {
      switch ( v21[32] & 0xF )
      {
        case 0:
        case 1:
        case 3:
        case 5:
        case 6:
        case 7:
        case 8:
          if ( (v21[80] & 2) != 0 )
            break;
          v7 = sub_14D81F0(*((_QWORD *)v21 - 3), a1);
          if ( !v7 )
            break;
          return v7;
        case 2:
        case 4:
        case 9:
        case 0xA:
          break;
        default:
          goto LABEL_119;
      }
    }
    v9 = *(_WORD *)(a1 + 18);
LABEL_13:
    if ( v9 == 47 )
    {
      v22 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      if ( *(_BYTE *)(*(_QWORD *)v22 + 8LL) == 15 )
      {
        v23 = (__int64 *)sub_14D8290(v22, **(_QWORD **)(*(_QWORD *)v22 + 16LL), a3);
        if ( v23 )
        {
          v7 = sub_14D66F0(v23, a2, (__int64)a3);
          if ( v7 )
            return v7;
        }
      }
    }
  }
  v41 = 0;
  v42 = 0;
  if ( (unsigned __int8)sub_14ACEF0(a1, &v41, 0, 1u) )
  {
    v10 = v42;
    if ( v42 )
    {
      v11 = sub_1643030(a2);
      if ( v11 >> 3 == v10 + 1 && (v11 & 7) == 0 )
      {
        v12 = *(_BYTE *)(a2 + 8);
        if ( v12 == 11 || (unsigned __int8)(v12 - 1) <= 5u )
        {
          v44 = v11;
          if ( v11 > 0x40 )
          {
            v40 = v11;
            sub_16A4EF0(&v43, 0, 0);
            v46 = v40;
            sub_16A4EF0(&v45, 0, 0);
            v10 = v42;
          }
          else
          {
            v43 = 0;
            v46 = v11;
            v45 = 0;
          }
          v13 = &v41[v10];
          v38 = v41;
          if ( !*a3 )
          {
            if ( v13 == v41 )
            {
LABEL_67:
              v24 = sub_16498A0(a1);
              v7 = sub_159C0E0(v24, &v43);
              if ( (unsigned __int8)(*(_BYTE *)(a2 + 8) - 1) <= 5u )
                v7 = sub_15A4510(v7, a2, 0);
              if ( v46 > 0x40 && v45 )
                j_j___libc_free_0_0(v45);
              if ( v44 > 0x40 && v43 )
                j_j___libc_free_0_0(v43);
              return v7;
            }
            while ( 1 )
            {
              v18 = *(v13 - 1);
              if ( v46 <= 0x40 )
              {
                v45 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v46) & v18;
              }
              else
              {
                *(_QWORD *)v45 = v18;
                memset((void *)(v45 + 8), 0, 8 * (unsigned int)(((unsigned __int64)v46 + 63) >> 6) - 8);
              }
              v14 = v44;
              v48 = v44;
              if ( v44 <= 0x40 )
                break;
              sub_16A4FD0(&v47, &v43);
              v14 = v48;
              if ( v48 <= 0x40 )
              {
                v15 = v44;
LABEL_28:
                v16 = 0;
                if ( v14 != 8 )
                  v16 = (v47 << 8) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v14);
                goto LABEL_30;
              }
              sub_16A7DC0(&v47, 8);
              v14 = v48;
              if ( v48 <= 0x40 )
              {
                v16 = v47;
                v15 = v44;
LABEL_30:
                v17 = v45 | v16;
                v47 = v17;
                goto LABEL_31;
              }
              sub_16A89F0(&v47, &v45);
              v14 = v48;
              v17 = v47;
              v15 = v44;
LABEL_31:
              v48 = 0;
              if ( v15 > 0x40 && v43 )
              {
                v35 = v14;
                v36 = v17;
                j_j___libc_free_0_0(v43);
                v43 = v36;
                v44 = v35;
                if ( v48 > 0x40 && v47 )
                  j_j___libc_free_0_0(v47);
              }
              else
              {
                v43 = v17;
                v44 = v14;
              }
              if ( --v13 == v38 )
                goto LABEL_67;
            }
            v15 = v44;
            v47 = v43;
            goto LABEL_28;
          }
          v25 = v41;
          if ( v13 == v41 )
          {
LABEL_96:
            if ( v46 > 0x40 )
            {
              *(_QWORD *)v45 = 0;
              memset((void *)(v45 + 8), 0, 8 * (unsigned int)(((unsigned __int64)v46 + 63) >> 6) - 8);
            }
            else
            {
              v45 = 0;
            }
            v31 = v44;
            v48 = v44;
            if ( v44 > 0x40 )
            {
              sub_16A4FD0(&v47, &v43);
              v31 = v48;
              if ( v48 > 0x40 )
              {
                sub_16A7DC0(&v47, 8);
                v31 = v48;
                if ( v48 > 0x40 )
                {
                  sub_16A89F0(&v47, &v45);
                  v31 = v48;
                  v34 = v47;
                  v32 = v44;
                  goto LABEL_103;
                }
                v33 = v47;
                v32 = v44;
LABEL_102:
                v34 = v45 | v33;
                v47 = v34;
LABEL_103:
                v48 = 0;
                if ( v32 > 0x40 && v43 )
                {
                  j_j___libc_free_0_0(v43);
                  v43 = v34;
                  v44 = v31;
                  if ( v48 > 0x40 && v47 )
                    j_j___libc_free_0_0(v47);
                }
                else
                {
                  v43 = v34;
                  v44 = v31;
                }
                goto LABEL_67;
              }
              v32 = v44;
            }
            else
            {
              v32 = v44;
              v47 = v43;
            }
            v33 = 0;
            if ( v31 != 8 )
              v33 = (v47 << 8) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v31);
            goto LABEL_102;
          }
          while ( 1 )
          {
            v30 = *v25;
            if ( v46 <= 0x40 )
            {
              v45 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v46) & v30;
            }
            else
            {
              *(_QWORD *)v45 = v30;
              memset((void *)(v45 + 8), 0, 8 * (unsigned int)(((unsigned __int64)v46 + 63) >> 6) - 8);
            }
            v26 = v44;
            v48 = v44;
            if ( v44 <= 0x40 )
              break;
            sub_16A4FD0(&v47, &v43);
            v26 = v48;
            if ( v48 <= 0x40 )
            {
              v27 = v44;
LABEL_81:
              v28 = 0;
              if ( v26 != 8 )
                v28 = (v47 << 8) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v26);
              goto LABEL_83;
            }
            sub_16A7DC0(&v47, 8);
            v26 = v48;
            if ( v48 <= 0x40 )
            {
              v28 = v47;
              v27 = v44;
LABEL_83:
              v29 = v45 | v28;
              v47 = v29;
              goto LABEL_84;
            }
            sub_16A89F0(&v47, &v45);
            v26 = v48;
            v29 = v47;
            v27 = v44;
LABEL_84:
            v48 = 0;
            if ( v27 > 0x40 && v43 )
            {
              v37 = v29;
              v39 = v26;
              j_j___libc_free_0_0(v43);
              v43 = v37;
              v44 = v39;
              if ( v48 > 0x40 && v47 )
                j_j___libc_free_0_0(v47);
            }
            else
            {
              v43 = v29;
              v44 = v26;
            }
            if ( v13 == ++v25 )
              goto LABEL_96;
          }
          v27 = v44;
          v47 = v43;
          goto LABEL_81;
        }
      }
    }
  }
  v19 = sub_14AD280(a1, (unsigned __int64)a3, 6u);
  v20 = v19;
  if ( *(_BYTE *)(v19 + 16) != 3 || (*(_BYTE *)(v19 + 80) & 1) == 0 || (unsigned __int8)sub_15E4F60(v19) )
    return sub_14D5F30(a1, a2, a3);
  switch ( *(_BYTE *)(v20 + 32) & 0xF )
  {
    case 0:
    case 1:
    case 3:
    case 5:
    case 6:
    case 7:
    case 8:
      if ( (*(_BYTE *)(v20 + 80) & 2) != 0 )
        return sub_14D5F30(a1, a2, a3);
      if ( (unsigned __int8)sub_1593BB0(*(_QWORD *)(v20 - 24)) )
      {
        v7 = sub_15A06D0(a2);
      }
      else
      {
        if ( *(_BYTE *)(*(_QWORD *)(v20 - 24) + 16LL) != 9 )
          return sub_14D5F30(a1, a2, a3);
        v7 = sub_1599EF0(a2);
      }
      break;
    case 2:
    case 4:
    case 9:
    case 0xA:
      return sub_14D5F30(a1, a2, a3);
    default:
      goto LABEL_119;
  }
  return v7;
}
