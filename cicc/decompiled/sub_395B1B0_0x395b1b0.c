// Function: sub_395B1B0
// Address: 0x395b1b0
//
__int64 __fastcall sub_395B1B0(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4, int *a5, unsigned int a6)
{
  __int64 **v8; // rax
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 *v11; // rbx
  unsigned int v12; // r14d
  __int64 *v13; // r12
  __int64 *v14; // r13
  __int64 *v15; // r10
  unsigned int v16; // ebx
  __int64 v17; // rsi
  __int64 v18; // r12
  __int64 *v19; // rbx
  __int64 v20; // r13
  __int64 v21; // rdi
  int v22; // esi
  __int64 v23; // rax
  __int64 v24; // rax
  _QWORD *v25; // rax
  _QWORD *v26; // rax
  __int64 v27; // r13
  __int64 result; // rax
  __int64 v29; // rax
  bool v30; // al
  char v31; // al
  _QWORD *v32; // rax
  __int64 v33; // rbx
  _QWORD *v34; // rax
  unsigned int v35; // eax
  __int64 v36; // rsi
  __int64 v37; // r8
  unsigned __int64 v38; // rcx
  int v39; // eax
  int v40; // eax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rsi
  __int64 v44; // rax
  __int64 v45; // rax
  unsigned int v46; // esi
  __int64 v47; // [rsp+20h] [rbp-E0h]
  __int64 v48; // [rsp+30h] [rbp-D0h]
  __int64 *v49; // [rsp+40h] [rbp-C0h]
  __int64 v50; // [rsp+48h] [rbp-B8h]
  __int64 v51; // [rsp+48h] [rbp-B8h]
  __int64 *v52; // [rsp+50h] [rbp-B0h]
  __int64 v53; // [rsp+50h] [rbp-B0h]
  __int64 *v54; // [rsp+50h] [rbp-B0h]
  __int64 *v55; // [rsp+50h] [rbp-B0h]
  __int64 v57; // [rsp+60h] [rbp-A0h]
  __int64 v58; // [rsp+68h] [rbp-98h]
  __int64 *v62; // [rsp+88h] [rbp-78h]
  __int64 v64; // [rsp+90h] [rbp-70h]
  __int64 v65; // [rsp+98h] [rbp-68h]
  __int64 v66; // [rsp+98h] [rbp-68h]
  unsigned int v67; // [rsp+A0h] [rbp-60h]
  __int64 v68; // [rsp+A0h] [rbp-60h]
  __int64 v70; // [rsp+A8h] [rbp-58h]
  __int64 *v71; // [rsp+A8h] [rbp-58h]
  __int64 v72[2]; // [rsp+B0h] [rbp-50h] BYREF
  __int16 v73; // [rsp+C0h] [rbp-40h]

  v62 = (__int64 *)sub_1644900(a1, a6);
  v67 = 0x20 / a6;
  v8 = (__int64 **)sub_16463B0(v62, 0x20 / a6);
  v9 = sub_1599EF0(v8);
  v10 = sub_1644900(a1, 0x20u);
  v11 = *(__int64 **)a4;
  v58 = v10;
  v65 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
  if ( *(_QWORD *)a4 != v65 )
  {
    v12 = 0;
    v72[0] = 0;
    v57 = v9;
    v13 = 0;
    v14 = v11;
    v15 = (__int64 *)sub_395AC60(a2, *v11, v72);
    v16 = 0;
    while ( v12 >> 3 == v72[0] )
    {
      if ( v13 )
      {
        ++v14;
        ++v16;
        if ( (__int64 *)v65 == v14 )
          goto LABEL_23;
      }
      else
      {
        v17 = *v15;
        v18 = 1;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v17 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v41 = *(_QWORD *)(v17 + 32);
              v17 = *(_QWORD *)(v17 + 24);
              v18 *= v41;
              continue;
            case 1:
              v29 = 16;
              break;
            case 2:
              v29 = 32;
              break;
            case 3:
            case 9:
              v29 = 64;
              break;
            case 4:
              v29 = 80;
              break;
            case 5:
            case 6:
              v29 = 128;
              break;
            case 7:
              v55 = v15;
              v40 = sub_15A9520(a2, 0);
              v15 = v55;
              v29 = (unsigned int)(8 * v40);
              break;
            case 0xB:
              v29 = *(_DWORD *)(v17 + 8) >> 8;
              break;
            case 0xD:
              v52 = v15;
              v34 = (_QWORD *)sub_15A9930(a2, v17);
              v15 = v52;
              v29 = 8LL * *v34;
              break;
            case 0xE:
              v49 = v15;
              v53 = *(_QWORD *)(v17 + 32);
              v50 = *(_QWORD *)(v17 + 24);
              v35 = sub_15A9FE0(a2, v50);
              v15 = v49;
              v36 = v50;
              v37 = 1;
              v38 = v35;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v36 + 8) )
                {
                  case 0:
                  case 8:
                  case 0xA:
                  case 0xC:
                  case 0x10:
                    v44 = *(_QWORD *)(v36 + 32);
                    v36 = *(_QWORD *)(v36 + 24);
                    v37 *= v44;
                    continue;
                  case 1:
                    v42 = 16;
                    goto LABEL_44;
                  case 2:
                    v42 = 32;
                    goto LABEL_44;
                  case 3:
                  case 9:
                    v42 = 64;
                    goto LABEL_44;
                  case 4:
                    v42 = 80;
                    goto LABEL_44;
                  case 5:
                  case 6:
                    v42 = 128;
                    goto LABEL_44;
                  case 7:
                    JUMPOUT(0x395B79E);
                  case 0xB:
                    v42 = *(_DWORD *)(v36 + 8) >> 8;
LABEL_44:
                    v29 = 8 * v38 * v53 * ((v38 + ((unsigned __int64)(v37 * v42 + 7) >> 3) - 1) / v38);
                    break;
                  case 0xD:
                    JUMPOUT(0x395B74F);
                  case 0xE:
                    v48 = *(_QWORD *)(v36 + 24);
                    sub_15A9FE0(a2, v48);
                    v51 = 1;
                    v43 = v48;
                    while ( 2 )
                    {
                      switch ( *(_BYTE *)(v43 + 8) )
                      {
                        case 0:
                        case 8:
                        case 0xA:
                        case 0xC:
                        case 0x10:
                          v45 = v51 * *(_QWORD *)(v43 + 32);
                          v43 = *(_QWORD *)(v43 + 24);
                          v51 = v45;
                          continue;
                        case 1:
                        case 2:
                        case 3:
                        case 4:
                        case 5:
                        case 6:
                        case 9:
                        case 0xB:
                          goto LABEL_55;
                        case 7:
                          v46 = 0;
                          goto LABEL_58;
                        case 0xD:
                          sub_15A9930(a2, v43);
                          goto LABEL_55;
                        case 0xE:
                          v47 = *(_QWORD *)(v43 + 24);
                          sub_15A9FE0(a2, v47);
                          sub_127FA20(a2, v47);
                          goto LABEL_55;
                        case 0xF:
                          v46 = *(_DWORD *)(v43 + 8) >> 8;
LABEL_58:
                          sub_15A9520(a2, v46);
LABEL_55:
                          JUMPOUT(0x395B812);
                      }
                    }
                  case 0xF:
                    JUMPOUT(0x395B7A5);
                }
                return result;
              }
            case 0xF:
              v54 = v15;
              v39 = sub_15A9520(a2, *(_DWORD *)(v17 + 8) >> 8);
              v15 = v54;
              v29 = (unsigned int)(8 * v39);
              break;
          }
          break;
        }
        if ( v29 * v18 != 32 )
          goto LABEL_10;
        v13 = v15;
        ++v14;
        ++v16;
        if ( (__int64 *)v65 == v14 )
        {
LABEL_23:
          v27 = (__int64)v13;
          v9 = v57;
          v30 = v67 == v16;
          goto LABEL_25;
        }
      }
      if ( v67 <= v16 )
        goto LABEL_23;
      v72[0] = 0;
      v12 += a6;
      v15 = (__int64 *)sub_395AC60(a2, *v14, v72);
      if ( v15 != v13 )
      {
LABEL_10:
        v9 = v57;
        goto LABEL_11;
      }
    }
    v27 = (__int64)v13;
    v9 = v57;
    v30 = v67 == v16 && v27 != 0;
LABEL_25:
    if ( v30 )
    {
      v31 = *(_BYTE *)(*(_QWORD *)v27 + 8LL);
      if ( v31 != 0 && v31 != 12 && (unsigned __int8)(v31 - 13) > 1u )
      {
        if ( v58 != *(_QWORD *)v27 )
        {
          v72[0] = (__int64)"base.bitcast";
          v73 = 259;
          v32 = sub_1648A60(56, 1u);
          v33 = (__int64)v32;
          if ( v32 )
            sub_15FD590((__int64)v32, v27, v58, (__int64)v72, 0);
          v27 = v33;
          sub_15F2180(v33, *a3);
          *a3 = v33;
        }
        return v27;
      }
    }
LABEL_11:
    v19 = *(__int64 **)a4;
    v64 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
    if ( *(_QWORD *)a4 != v64 )
    {
      v20 = 0;
      do
      {
        v21 = *v19;
        v72[0] = (__int64)"vec.elem";
        v22 = *a5;
        v73 = 259;
        v70 = v20;
        v66 = sub_3958FF0(v21, v22, (__int64)v62, a3, (__int64)v72);
        ++v20;
        v73 = 257;
        v23 = sub_1644900(a1, 0x20u);
        v24 = sub_159C470(v23, v70, 0);
        v71 = (__int64 *)v9;
        v68 = v24;
        v25 = sub_1648A60(56, 3u);
        v9 = (__int64)v25;
        if ( v25 )
          sub_15FA480((__int64)v25, v71, v66, v68, (__int64)v72, 0);
        ++v19;
        sub_15F2180(v9, *a3);
        *a3 = v9;
      }
      while ( v19 != (__int64 *)v64 );
    }
  }
  v72[0] = (__int64)"vec";
  v73 = 259;
  v26 = sub_1648A60(56, 1u);
  v27 = (__int64)v26;
  if ( v26 )
    sub_15FD590((__int64)v26, v9, v58, (__int64)v72, 0);
  sub_15F2180(v27, *a3);
  *a3 = v27;
  return v27;
}
