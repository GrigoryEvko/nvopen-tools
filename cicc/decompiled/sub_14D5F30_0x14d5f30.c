// Function: sub_14D5F30
// Address: 0x14d5f30
//
__int64 __fastcall sub_14D5F30(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 v4; // r12
  char v5; // al
  unsigned int v6; // ebx
  __int64 v7; // r14
  __int64 v8; // r15
  unsigned int v9; // r14d
  __int64 result; // rax
  __int64 v11; // rbx
  unsigned __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r14
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r9
  unsigned __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // r9
  unsigned __int64 v26; // rsi
  char *v27; // rdx
  __int64 v28; // rcx
  unsigned int v29; // eax
  bool v30; // zf
  __int64 v31; // rsi
  int v32; // r14d
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rcx
  __int64 v35; // rdx
  __int64 v36; // r13
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rsi
  int v40; // eax
  __int64 v41; // rax
  _QWORD *v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rsi
  char *v45; // r13
  char *v46; // rbx
  unsigned __int64 v47; // rdx
  unsigned __int64 v48; // rcx
  __int64 v49; // rax
  unsigned __int64 v50; // [rsp+0h] [rbp-A0h]
  unsigned int v51; // [rsp+8h] [rbp-98h]
  __int64 v52; // [rsp+8h] [rbp-98h]
  __int64 v53; // [rsp+10h] [rbp-90h]
  unsigned __int64 v54; // [rsp+10h] [rbp-90h]
  unsigned __int64 v55; // [rsp+10h] [rbp-90h]
  unsigned __int64 v56; // [rsp+10h] [rbp-90h]
  unsigned __int64 v57; // [rsp+10h] [rbp-90h]
  __int64 v58; // [rsp+18h] [rbp-88h]
  __int64 v59; // [rsp+18h] [rbp-88h]
  __int64 v60; // [rsp+18h] [rbp-88h]
  __int64 v61; // [rsp+18h] [rbp-88h]
  __int64 v62; // [rsp+18h] [rbp-88h]
  __int64 v63; // [rsp+28h] [rbp-78h] BYREF
  __int64 *v64; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v65; // [rsp+38h] [rbp-68h]
  unsigned __int64 v66; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v67; // [rsp+48h] [rbp-58h]
  _OWORD v68[5]; // [rsp+50h] [rbp-50h] BYREF

  v4 = a2;
  v5 = *(_BYTE *)(a2 + 8);
  if ( v5 != 11 )
  {
    v9 = *(_DWORD *)(*(_QWORD *)a1 + 8LL) >> 8;
    switch ( v5 )
    {
      case 1:
        v13 = sub_16498A0(a1);
        v14 = sub_1643340(v13);
        break;
      case 2:
        v19 = sub_16498A0(a1);
        v14 = sub_1643350(v19);
        break;
      case 3:
        v20 = sub_16498A0(a1);
        v14 = sub_1643360(v20);
        break;
      case 16:
        v11 = 1;
        v12 = (unsigned int)sub_15A9FE0(a3, a2);
        while ( 2 )
        {
          switch ( *(_BYTE *)(a2 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v38 = *(_QWORD *)(a2 + 32);
              a2 = *(_QWORD *)(a2 + 24);
              v11 *= v38;
              continue;
            case 1:
              v60 = 16;
              goto LABEL_61;
            case 2:
              v60 = 32;
              goto LABEL_61;
            case 3:
            case 9:
              v60 = 64;
              goto LABEL_61;
            case 4:
              v60 = 80;
              goto LABEL_61;
            case 5:
            case 6:
              v60 = 128;
              goto LABEL_61;
            case 7:
              v55 = v12;
              v39 = 0;
              goto LABEL_68;
            case 0xB:
              v60 = *(_DWORD *)(a2 + 8) >> 8;
              goto LABEL_61;
            case 0xD:
              v57 = v12;
              v42 = (_QWORD *)sub_15A9930(a3, a2);
              v12 = v57;
              v60 = 8LL * *v42;
              goto LABEL_61;
            case 0xE:
              v50 = v12;
              v52 = *(_QWORD *)(a2 + 24);
              v61 = *(_QWORD *)(a2 + 32);
              v56 = (unsigned int)sub_15A9FE0(a3, v52);
              v41 = sub_127FA20((__int64)a3, v52);
              v12 = v50;
              v60 = 8 * v56 * v61 * ((v56 + ((unsigned __int64)(v41 + 7) >> 3) - 1) / v56);
              goto LABEL_61;
            case 0xF:
              v55 = v12;
              v39 = *(_DWORD *)(a2 + 8) >> 8;
LABEL_68:
              v40 = sub_15A9520(a3, v39);
              v12 = v55;
              v60 = (unsigned int)(8 * v40);
LABEL_61:
              v54 = v12;
              v37 = sub_16498A0(a1);
              v14 = sub_1644C60(v37, 8 * v54 * ((v54 + ((unsigned __int64)(v60 * v11 + 7) >> 3) - 1) / v54));
              break;
          }
          break;
        }
        break;
      default:
        return 0;
    }
    v15 = sub_1647190(v14, v9);
    if ( !(unsigned __int8)sub_1593BB0(a1) || *(_BYTE *)(v15 + 8) == 9 )
      v16 = sub_14D44C0(a1, v15, a3);
    else
      v16 = sub_15A06D0(v15);
    v17 = sub_14D5F30(v16, v14, a3);
    v18 = v17;
    if ( v17 )
    {
      if ( !(unsigned __int8)sub_1593BB0(v17) || *(_BYTE *)(v4 + 8) == 9 )
        return sub_14D44C0(v18, v4, a3);
      else
        return sub_15A06D0(v4);
    }
    return 0;
  }
  v6 = (unsigned int)((*(_DWORD *)(a2 + 8) >> 8) + 7) >> 3;
  v7 = v6 - 1;
  if ( (unsigned int)v7 > 0x1F )
    return 0;
  v65 = 1;
  v64 = 0;
  if ( (unsigned __int8)sub_14D5D40(a1, &v63, (__int64)&v64, (__int64)a3)
    && (v8 = v63, *(_BYTE *)(v63 + 16) == 3)
    && (*(_BYTE *)(v63 + 80) & 1) != 0
    && !(unsigned __int8)sub_15E4F60(v63) )
  {
    switch ( *(_BYTE *)(v8 + 32) & 0xF )
    {
      case 0:
      case 1:
      case 3:
      case 5:
      case 6:
      case 7:
      case 8:
        if ( (*(_BYTE *)(v8 + 80) & 2) != 0 )
          goto LABEL_14;
        v21 = **(_QWORD **)(v8 - 24);
        v22 = *(unsigned __int8 *)(v21 + 8);
        if ( (unsigned __int8)v22 <= 0xFu )
        {
          v43 = 35454;
          if ( _bittest64(&v43, v22) )
            goto LABEL_38;
        }
        if ( (unsigned int)(v22 - 13) > 1 && (_DWORD)v22 != 16
          || !(unsigned __int8)sub_16435F0(**(_QWORD **)(v8 - 24), 0) )
        {
          goto LABEL_14;
        }
        v21 = **(_QWORD **)(v8 - 24);
LABEL_38:
        if ( v65 > 0x40 )
          v23 = *v64;
        else
          v23 = (__int64)((_QWORD)v64 << (64 - (unsigned __int8)v65)) >> (64 - (unsigned __int8)v65);
        v53 = v23;
        v59 = v21;
        v51 = sub_15A9FE0(a3, v21);
        v24 = sub_127FA20((__int64)a3, v59);
        if ( v53 + v6 <= 0 || (__int64)(v51 * ((v51 + ((unsigned __int64)(v24 + 7) >> 3) - 1) / v51)) <= v53 )
        {
          result = sub_1599EF0(a2);
          break;
        }
        memset(v68, 0, 32);
        if ( v53 < 0 )
        {
          v26 = 0;
          v28 = v6 + (unsigned int)v53;
          v27 = (char *)v68 - v53;
        }
        else
        {
          v26 = v53;
          v27 = (char *)v68;
          v28 = v6;
        }
        if ( !(unsigned __int8)sub_14D5510(*(_QWORD *)(v8 - 24), v26, (__int64)v27, v28, a3, v25) )
          goto LABEL_14;
        v29 = *(_DWORD *)(v4 + 8) >> 8;
        v67 = v29;
        if ( v29 > 0x40 )
        {
          sub_16A4EF0(&v66, 0, 0);
          if ( !*a3 )
          {
            LOBYTE(v29) = v67;
            v31 = *((unsigned __int8 *)v68 + v7);
            if ( v67 > 0x40 )
            {
              *(_QWORD *)v66 = v31;
              memset((void *)(v66 + 8), 0, 8 * (unsigned int)(((unsigned __int64)v67 + 63) >> 6) - 8);
LABEL_49:
              v32 = 2;
              if ( v6 == 1 )
                goto LABEL_77;
              while ( 1 )
              {
                v36 = v6 - v32;
                if ( v67 <= 0x40 )
                  break;
                sub_16A7DC0(&v66, 8);
                v35 = *((unsigned __int8 *)v68 + v36);
                if ( v67 <= 0x40 )
                {
                  v33 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v67;
LABEL_54:
                  v66 = v33 & (v66 | v35);
                  goto LABEL_55;
                }
                *(_QWORD *)v66 |= v35;
LABEL_55:
                if ( v6 == v32 )
                  goto LABEL_77;
                ++v32;
              }
              v33 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v67;
              v34 = 0;
              if ( v67 != 8 )
                v34 = v33 & (v66 << 8);
              v66 = v34;
              v35 = *((unsigned __int8 *)v68 + v36);
              goto LABEL_54;
            }
LABEL_48:
            v66 = v31 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v29);
            goto LABEL_49;
          }
          LOBYTE(v29) = v67;
          v44 = LOBYTE(v68[0]);
          if ( v67 > 0x40 )
          {
            *(_QWORD *)v66 = LOBYTE(v68[0]);
            memset((void *)(v66 + 8), 0, 8 * (unsigned int)(((unsigned __int64)v67 + 63) >> 6) - 8);
            goto LABEL_84;
          }
        }
        else
        {
          v30 = *a3 == 0;
          v66 = 0;
          if ( v30 )
          {
            v31 = *((unsigned __int8 *)v68 + v7);
            goto LABEL_48;
          }
          v44 = LOBYTE(v68[0]);
        }
        v66 = v44 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v29);
LABEL_84:
        if ( v6 != 1 )
        {
          v45 = (char *)v68 + 1;
          v46 = (char *)v68 + v6;
          do
          {
            if ( v67 <= 0x40 )
            {
              v47 = 0;
              v48 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v67;
              if ( v67 != 8 )
                v47 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v67) & (v66 << 8);
              v66 = v47;
              v49 = (unsigned __int8)*v45;
            }
            else
            {
              sub_16A7DC0(&v66, 8);
              v49 = (unsigned __int8)*v45;
              if ( v67 > 0x40 )
              {
                *(_QWORD *)v66 |= v49;
                goto LABEL_90;
              }
              v48 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v67;
            }
            v66 = v48 & (v66 | v49);
LABEL_90:
            ++v45;
          }
          while ( v46 != v45 );
        }
LABEL_77:
        result = sub_159C0E0(*(_QWORD *)v4, &v66);
        if ( v67 > 0x40 && v66 )
        {
          v62 = result;
          j_j___libc_free_0_0(v66);
          result = v62;
        }
        break;
      case 2:
      case 4:
      case 9:
      case 0xA:
        goto LABEL_14;
      default:
        JUMPOUT(0x419798);
    }
  }
  else
  {
LABEL_14:
    result = 0;
  }
  if ( v65 > 0x40 )
  {
    if ( v64 )
    {
      v58 = result;
      j_j___libc_free_0_0(v64);
      return v58;
    }
  }
  return result;
}
