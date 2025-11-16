// Function: sub_14A4C90
// Address: 0x14a4c90
//
__int64 __fastcall sub_14A4C90(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  __int64 *v4; // rbx
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v7; // r14
  unsigned int v8; // ebx
  unsigned __int8 *v9; // rax
  unsigned __int8 *v10; // rcx
  unsigned int v11; // eax
  __int64 v12; // rdi
  unsigned int v13; // r13d
  unsigned __int8 *v14; // rax
  unsigned __int8 *v15; // rcx
  __int64 v16; // rsi
  unsigned __int8 *v17; // rax
  __int64 v18; // rsi
  unsigned __int8 *v19; // rcx
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // r12
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int8 v28; // dl
  unsigned __int64 v29; // rax
  int v30; // eax
  __int64 result; // rax
  unsigned __int64 v32; // rcx
  _BYTE *v33; // r15
  unsigned __int64 v34; // rax
  _QWORD *v35; // r8
  _QWORD *v36; // rbx
  __int64 v37; // rcx
  _QWORD *v38; // rax
  unsigned __int64 v39; // r13
  int v40; // edx
  __int64 v41; // rbx
  unsigned __int64 v42; // rax
  int v43; // eax
  __int64 v44; // rdi
  __int64 **v45; // rax
  __int64 v46; // r9
  __int64 v47; // rcx
  int v48; // r8d
  __int64 *v49; // rdi
  unsigned int v50; // [rsp+8h] [rbp-98h]
  _QWORD *v51; // [rsp+8h] [rbp-98h]
  unsigned __int64 v52; // [rsp+18h] [rbp-88h] BYREF
  _BYTE *v53; // [rsp+20h] [rbp-80h] BYREF
  __int64 v54; // [rsp+28h] [rbp-78h]
  _BYTE v55[40]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v56; // [rsp+58h] [rbp-48h]
  __int64 v57; // [rsp+60h] [rbp-40h]
  __int64 v58; // [rsp+68h] [rbp-38h]
  __int64 savedregs; // [rsp+A0h] [rbp+0h] BYREF

  v28 = *(_BYTE *)(a2 + 16);
  if ( v28 <= 0x17u )
  {
    if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 1 )
    {
      v46 = *(_QWORD *)a2;
      v47 = 0;
      goto LABEL_33;
    }
    v44 = *(_QWORD *)a2;
LABEL_26:
    v45 = (__int64 **)(a2 - 24);
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v45 = *(__int64 ***)(a2 - 8);
    v46 = v44;
    v47 = **v45;
    if ( v28 > 0x17u )
    {
      v30 = v28;
LABEL_30:
      v48 = v30 - 24;
LABEL_31:
      v49 = (__int64 *)a1;
      savedregs = (__int64)&savedregs;
      v2 = v47;
      v3 = v46;
      v4 = v49;
      switch ( v48 )
      {
        case 17:
        case 18:
        case 19:
        case 20:
        case 21:
        case 22:
          return 4;
        case 23:
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
        case 32:
        case 33:
        case 34:
        case 35:
        case 37:
        case 38:
        case 39:
        case 40:
        case 41:
        case 42:
        case 43:
        case 44:
          return 1;
        case 36:
          v5 = *v49;
          v6 = 1;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v3 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v24 = *(_QWORD *)(v3 + 32);
                v3 = *(_QWORD *)(v3 + 24);
                v6 *= v24;
                continue;
              case 1:
                v16 = 16;
                goto LABEL_59;
              case 2:
                v16 = 32;
                goto LABEL_59;
              case 3:
              case 9:
                v16 = 64;
                goto LABEL_59;
              case 4:
                v16 = 80;
                goto LABEL_59;
              case 5:
              case 6:
                v16 = 128;
                goto LABEL_59;
              case 7:
                v16 = 8 * (unsigned int)sub_15A9520(*v49, 0);
                goto LABEL_59;
              case 0xB:
                v16 = *(_DWORD *)(v3 + 8) >> 8;
                goto LABEL_59;
              case 0xD:
                v16 = 8LL * *(_QWORD *)sub_15A9930(*v49, v3);
                goto LABEL_59;
              case 0xE:
                v20 = *(_QWORD *)(v3 + 24);
                v21 = *(_QWORD *)(v3 + 32);
                v22 = *v49;
                v23 = 1;
                v58 = v21;
                sub_15A9FE0(v22, v20);
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v20 + 8) )
                  {
                    case 0:
                    case 8:
                    case 0xA:
                    case 0xC:
                    case 0x10:
                      v25 = *(_QWORD *)(v20 + 32);
                      v20 = *(_QWORD *)(v20 + 24);
                      v23 *= v25;
                      continue;
                    case 1:
                    case 2:
                    case 4:
                    case 5:
                    case 6:
                    case 0xB:
                      goto LABEL_83;
                    case 3:
                    case 9:
                      JUMPOUT(0x14A1F62);
                    case 7:
                      sub_15A9520(v5, 0);
                      goto LABEL_83;
                    case 0xD:
                      sub_15A9930(v5, v20);
                      goto LABEL_83;
                    case 0xE:
                      v26 = *(_QWORD *)(v20 + 32);
                      v56 = *(_QWORD *)(v20 + 24);
                      v57 = v26;
                      sub_15A9FE0(v5, v56);
                      sub_127FA20(v5, v56);
                      goto LABEL_83;
                    case 0xF:
                      sub_15A9520(v5, *(_DWORD *)(v20 + 8) >> 8);
LABEL_83:
                      JUMPOUT(0x14A1F70);
                  }
                }
              case 0xF:
                v16 = 8 * (unsigned int)sub_15A9520(*v49, *(_DWORD *)(v3 + 8) >> 8);
LABEL_59:
                v17 = *(unsigned __int8 **)(v5 + 24);
                v18 = v6 * v16;
                v19 = &v17[*(unsigned int *)(v5 + 32)];
                if ( v17 == v19 )
                  return 1;
                break;
            }
            break;
          }
          while ( *v17 != v18 )
          {
            if ( v19 == ++v17 )
              return 1;
          }
          return 0;
        case 45:
          v7 = *v49;
          v8 = sub_16431D0(v46);
          v9 = *(unsigned __int8 **)(v7 + 24);
          v10 = &v9[*(unsigned int *)(v7 + 32)];
          if ( v9 == v10 )
            return 1;
          while ( v8 != *v9 )
          {
            if ( v10 == ++v9 )
              return 1;
          }
          return v8 < (unsigned int)sub_15A9570(v7, v2);
        case 46:
          v11 = sub_16431D0(v47);
          v12 = *v49;
          v13 = v11;
          v14 = *(unsigned __int8 **)(*v4 + 24);
          v15 = &v14[*(unsigned int *)(*v4 + 32)];
          if ( v14 == v15 )
            return 1;
          break;
        case 47:
          if ( v46 == v47 )
            return 0;
          return *(_BYTE *)(v46 + 8) != 15 || *(_BYTE *)(v47 + 8) != 15;
        default:
          return 1;
      }
      while ( v13 != *v14 )
      {
        if ( v15 == ++v14 )
          return 1;
      }
      return (unsigned int)sub_15A9570(v12, v3) < v13;
    }
LABEL_33:
    v48 = 56;
    if ( v28 == 5 )
      v48 = *(unsigned __int16 *)(a2 + 18);
    goto LABEL_31;
  }
  v29 = a2 | 4;
  if ( v28 != 78 && (v29 = a2 & 0xFFFFFFFFFFFFFFFBLL, v28 != 29)
    || (v52 = v29, v32 = v29 & 0xFFFFFFFFFFFFFFF8LL, (v29 & 0xFFFFFFFFFFFFFFF8LL) == 0) )
  {
    v30 = v28;
    if ( (unsigned int)v28 - 60 <= 0xC )
    {
      if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(a2 - 24) + 16LL) - 75) <= 1u )
        return 0;
      if ( (unsigned __int8)(v28 - 61) <= 1u || v28 == 68 )
        return 1;
    }
    v46 = *(_QWORD *)a2;
    v44 = *(_QWORD *)a2;
    if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 1 )
    {
      v47 = 0;
      goto LABEL_30;
    }
    goto LABEL_26;
  }
  if ( (v29 & 4) != 0 )
  {
    v33 = *(_BYTE **)(v32 - 24);
    if ( !v33[16] )
    {
LABEL_11:
      v34 = sub_134EF80(&v52);
      v53 = v55;
      v35 = (_QWORD *)v34;
      v36 = (_QWORD *)((v52 & 0xFFFFFFFFFFFFFFF8LL) - 24LL
                                                    * (*(_DWORD *)((v52 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
      v37 = v34 - (_QWORD)v36;
      v54 = 0x800000000LL;
      v38 = v55;
      v39 = 0xAAAAAAAAAAAAAAABLL * (v37 >> 3);
      v40 = 0;
      if ( (unsigned __int64)v37 > 0xC0 )
      {
        v51 = v35;
        sub_16CD150(&v53, v55, 0xAAAAAAAAAAAAAAABLL * (v37 >> 3), 8);
        v40 = v54;
        v35 = v51;
        v38 = &v53[8 * (unsigned int)v54];
      }
      if ( v35 != v36 )
      {
        do
        {
          if ( v38 )
            *v38 = *v36;
          v36 += 3;
          ++v38;
        }
        while ( v35 != v36 );
        v40 = v54;
      }
      LODWORD(v54) = v39 + v40;
      result = sub_14A2470(a1, (__int64)v33, (int)v39 + v40);
      if ( v53 != v55 )
      {
        v50 = result;
        _libc_free((unsigned __int64)v53);
        return v50;
      }
      return result;
    }
  }
  else
  {
    v33 = *(_BYTE **)(v32 - 72);
    if ( !v33[16] )
      goto LABEL_11;
  }
  v41 = **(_QWORD **)(*(_QWORD *)v33 + 16LL);
  v42 = sub_134EF80(&v52);
  v43 = -1431655765
      * ((__int64)(v42
                 - ((v52 & 0xFFFFFFFFFFFFFFF8LL)
                  - 24LL * (*(_DWORD *)((v52 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF))) >> 3);
  if ( v43 < 0 )
    v43 = *(_DWORD *)(v41 + 12) - 1;
  return (unsigned int)(v43 + 1);
}
