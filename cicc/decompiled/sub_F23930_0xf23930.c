// Function: sub_F23930
// Address: 0xf23930
//
__int64 __fastcall sub_F23930(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  _QWORD *v3; // r13
  __int64 v5; // rax
  int v6; // r15d
  __int64 v8; // rdi
  __int64 v9; // r13
  __int64 v10; // rax
  int v11; // r15d
  unsigned int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 v16; // r15
  _QWORD *v17; // rax
  __int64 *v18; // r14
  __int64 v19; // rbx
  __int64 v20; // rax
  __int16 v21; // r14
  __int64 v22; // rbx
  _QWORD **v23; // rdx
  int v24; // ecx
  int v25; // eax
  __int64 *v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rdx
  _BYTE *v29; // rax
  unsigned int v30; // edx
  unsigned int v31; // ebx
  __int64 v32; // rsi
  unsigned __int64 v33; // rax
  __int64 v34; // rax
  int v35; // r13d
  __int64 v36; // r15
  __int64 v37; // rbx
  __int64 v38; // rax
  unsigned __int64 v39; // rax
  __int64 v40; // rbx
  __int64 v41; // r12
  _QWORD **v42; // rdx
  int v43; // ecx
  int v44; // eax
  __int64 *v45; // rax
  __int64 v46; // rsi
  __int64 v47; // rdi
  __int64 v48; // rbx
  unsigned int v49; // eax
  __int64 v50; // rax
  __int64 v51; // r12
  _QWORD **v52; // rax
  int v53; // ecx
  int v54; // edx
  __int64 *v55; // rax
  __int64 v56; // rsi
  __int64 v57; // rbx
  __int64 v58; // r12
  __int64 v59; // rdx
  unsigned int v60; // esi
  unsigned int v61; // [rsp+8h] [rbp-E8h]
  int v62; // [rsp+8h] [rbp-E8h]
  __int64 v63; // [rsp+8h] [rbp-E8h]
  __int64 v64; // [rsp+8h] [rbp-E8h]
  int v65; // [rsp+1Ch] [rbp-D4h] BYREF
  __int64 v66; // [rsp+20h] [rbp-D0h] BYREF
  unsigned int v67; // [rsp+28h] [rbp-C8h]
  _QWORD *v68; // [rsp+30h] [rbp-C0h] BYREF
  unsigned int v69; // [rsp+38h] [rbp-B8h]
  __int64 v70; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v71; // [rsp+48h] [rbp-A8h]
  __int64 v72; // [rsp+50h] [rbp-A0h]
  unsigned int v73; // [rsp+58h] [rbp-98h]
  __int64 v74[4]; // [rsp+60h] [rbp-90h] BYREF
  __int16 v75; // [rsp+80h] [rbp-70h]
  _BYTE v76[32]; // [rsp+90h] [rbp-60h] BYREF
  __int16 v77; // [rsp+B0h] [rbp-40h]

  v2 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v2 != 85 )
    return 0;
  v5 = *(_QWORD *)(v2 - 32);
  if ( !v5 || *(_BYTE *)v5 || *(_QWORD *)(v5 + 24) != *(_QWORD *)(v2 + 80) || (*(_BYTE *)(v5 + 33) & 0x20) == 0 )
    return 0;
  v6 = *(_DWORD *)(v5 + 36);
  if ( v6 == 312 )
  {
LABEL_10:
    v8 = *(_QWORD *)(v2 + 32 * (1LL - (*(_DWORD *)(v2 + 4) & 0x7FFFFFF)));
    v9 = v8 + 24;
    if ( *(_BYTE *)v8 != 17 )
    {
      v28 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v8 + 8) + 8LL) - 17;
      if ( (unsigned int)v28 > 1
        || *(_BYTE *)v8 > 0x15u
        || (v29 = sub_AD7630(v8, 1, v28)) == 0
        || (v9 = (__int64)(v29 + 24), *v29 != 17) )
      {
        v9 = 0;
        goto LABEL_14;
      }
    }
    if ( **(_DWORD **)(a2 + 72) || v6 != 333 && v6 != 369 )
      goto LABEL_14;
    v30 = *(_DWORD *)(v9 + 8);
    if ( v30 )
    {
      if ( v30 <= 0x40 )
      {
        v39 = *(_QWORD *)v9;
        if ( *(_QWORD *)v9 != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v30) )
        {
          if ( !v39 || (v39 & (v39 - 1)) != 0 )
            goto LABEL_14;
          goto LABEL_51;
        }
      }
      else
      {
        v62 = *(_DWORD *)(v9 + 8);
        if ( v62 != (unsigned int)sub_C445E0(v9) )
        {
          if ( (unsigned int)sub_C44630(v9) != 1 )
          {
LABEL_14:
            v10 = *(_QWORD *)(v2 + 16);
            if ( v10 && !*(_QWORD *)(v10 + 8) )
            {
              if ( !**(_DWORD **)(a2 + 72) )
              {
                v35 = sub_B5B5E0(v2);
                v36 = *(_QWORD *)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF));
                v37 = *(_QWORD *)(v2 + 32 * (1LL - (*(_DWORD *)(v2 + 4) & 0x7FFFFFF)));
                v38 = sub_ACADE0(*(__int64 ***)(v2 + 8));
                sub_F162A0(a1, v2, v38);
                sub_F207A0(a1, (__int64 *)v2);
                v77 = 257;
                return sub_B504D0(v35, v36, v37, (__int64)v76, 0, 0);
              }
              switch ( v6 )
              {
                case 372:
                  v40 = *(_QWORD *)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF));
                  v41 = *(_QWORD *)(v2 + 32 * (1LL - (*(_DWORD *)(v2 + 4) & 0x7FFFFFF)));
                  v77 = 257;
                  v3 = sub_BD2C40(72, unk_3F10FD0);
                  if ( v3 )
                  {
                    v42 = *(_QWORD ***)(v40 + 8);
                    v43 = *((unsigned __int8 *)v42 + 8);
                    if ( (unsigned int)(v43 - 17) > 1 )
                    {
                      v46 = sub_BCB2A0(*v42);
                    }
                    else
                    {
                      v44 = *((_DWORD *)v42 + 8);
                      BYTE4(v68) = (_BYTE)v43 == 18;
                      LODWORD(v68) = v44;
                      v45 = (__int64 *)sub_BCB2A0(*v42);
                      v46 = sub_BCE1B0(v45, (__int64)v68);
                    }
                    sub_B523C0((__int64)v3, v46, 53, 36, v40, v41, (__int64)v76, 0, 0, 0);
                  }
                  return (__int64)v3;
                case 333:
                  v47 = *(_QWORD *)(*(_QWORD *)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF)) + 8LL);
                  if ( (unsigned int)*(unsigned __int8 *)(v47 + 8) - 17 <= 1 )
                    v47 = **(_QWORD **)(v47 + 16);
                  if ( sub_BCAC40(v47, 1) )
                  {
                    v77 = 257;
                    return sub_B504D0(
                             28,
                             *(_QWORD *)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF)),
                             *(_QWORD *)(v2 + 32 * (1LL - (*(_DWORD *)(v2 + 4) & 0x7FFFFFF))),
                             (__int64)v76,
                             0,
                             0);
                  }
                  break;
                case 369:
                  v48 = *(_QWORD *)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF));
                  if ( *(_QWORD *)(v2 + 32 * (1LL - (*(_DWORD *)(v2 + 4) & 0x7FFFFFF))) == v48 )
                  {
                    v49 = sub_BCB060(*(_QWORD *)(v48 + 8));
                    if ( (v49 & 1) == 0 )
                    {
                      sub_F0A5D0((__int64)v74, v49, v49 >> 1);
                      v50 = sub_AD8D80(
                              *(_QWORD *)(*(_QWORD *)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF)) + 8LL),
                              (__int64)v74);
                      v77 = 257;
                      v51 = v50;
                      v3 = sub_BD2C40(72, unk_3F10FD0);
                      if ( v3 )
                      {
                        v52 = *(_QWORD ***)(v48 + 8);
                        v53 = *((unsigned __int8 *)v52 + 8);
                        if ( (unsigned int)(v53 - 17) > 1 )
                        {
                          v56 = sub_BCB2A0(*v52);
                        }
                        else
                        {
                          v54 = *((_DWORD *)v52 + 8);
                          BYTE4(v70) = (_BYTE)v53 == 18;
                          LODWORD(v70) = v54;
                          v55 = (__int64 *)sub_BCB2A0(*v52);
                          v56 = sub_BCE1B0(v55, v70);
                        }
                        sub_B523C0((__int64)v3, v56, 53, 34, v48, v51, (__int64)v76, 0, 0, 0);
                      }
                      sub_969240(v74);
                      return (__int64)v3;
                    }
                  }
                  break;
              }
              if ( v9 )
              {
                v11 = sub_B5B690(v2);
                v12 = sub_B5B5E0(v2);
                sub_AB3450((__int64)&v70, v12, v9, v11);
                v67 = 1;
                v66 = 0;
                v69 = 1;
                v68 = 0;
                sub_AAF830((__int64)&v70, &v65, (__int64)&v66, (__int64 *)&v68);
                v13 = *(_DWORD *)(v2 + 4) & 0x7FFFFFF;
                v14 = *(_QWORD *)(v2 + 32 * (1 - v13));
                v15 = *(_QWORD *)(v2 - 32 * v13);
                v16 = *(_QWORD *)(v14 + 8);
                v61 = v69;
                if ( v69 > 0x40 )
                {
                  if ( v61 - (unsigned int)sub_C444A0((__int64)&v68) > 0x40 )
                    goto LABEL_24;
                  v17 = (_QWORD *)*v68;
                }
                else
                {
                  v17 = v68;
                }
                if ( !v17 )
                {
LABEL_26:
                  v21 = sub_B52870(v65);
                  v22 = sub_AD8D80(v16, (__int64)&v66);
                  v77 = 257;
                  v3 = sub_BD2C40(72, unk_3F10FD0);
                  if ( v3 )
                  {
                    v23 = *(_QWORD ***)(v15 + 8);
                    v24 = *((unsigned __int8 *)v23 + 8);
                    if ( (unsigned int)(v24 - 17) > 1 )
                    {
                      v27 = sub_BCB2A0(*v23);
                    }
                    else
                    {
                      v25 = *((_DWORD *)v23 + 8);
                      BYTE4(v74[0]) = (_BYTE)v24 == 18;
                      LODWORD(v74[0]) = v25;
                      v26 = (__int64 *)sub_BCB2A0(*v23);
                      v27 = sub_BCE1B0(v26, v74[0]);
                    }
                    sub_B523C0((__int64)v3, v27, 53, v21, v15, v22, (__int64)v76, 0, 0, 0);
                  }
                  if ( v69 > 0x40 && v68 )
                    j_j___libc_free_0_0(v68);
                  if ( v67 > 0x40 && v66 )
                    j_j___libc_free_0_0(v66);
                  if ( v73 > 0x40 && v72 )
                    j_j___libc_free_0_0(v72);
                  if ( v71 > 0x40 && v70 )
                    j_j___libc_free_0_0(v70);
                  return (__int64)v3;
                }
LABEL_24:
                v18 = *(__int64 **)(a1 + 32);
                v75 = 257;
                v19 = sub_AD8D80(v16, (__int64)&v68);
                v20 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v18[10] + 32LL))(
                        v18[10],
                        13,
                        v15,
                        v19,
                        0,
                        0);
                if ( !v20 )
                {
                  v77 = 257;
                  v63 = sub_B504D0(13, v15, v19, (__int64)v76, 0, 0);
                  (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)v18[11] + 16LL))(
                    v18[11],
                    v63,
                    v74,
                    v18[7],
                    v18[8]);
                  v57 = *v18;
                  v20 = v63;
                  v58 = *v18 + 16LL * *((unsigned int *)v18 + 2);
                  if ( *v18 != v58 )
                  {
                    do
                    {
                      v59 = *(_QWORD *)(v57 + 8);
                      v60 = *(_DWORD *)v57;
                      v57 += 16;
                      v64 = v20;
                      sub_B99FD0(v20, v60, v59);
                      v20 = v64;
                    }
                    while ( v58 != v57 );
                  }
                }
                v15 = v20;
                goto LABEL_26;
              }
            }
            return 0;
          }
LABEL_51:
          v77 = 257;
          v31 = *(_DWORD *)(v9 + 8);
          if ( v31 > 0x40 )
          {
            v32 = v31 - 1 - (unsigned int)sub_C444A0(v9);
          }
          else
          {
            v32 = 0xFFFFFFFFLL;
            if ( *(_QWORD *)v9 )
            {
              _BitScanReverse64(&v33, *(_QWORD *)v9);
              v32 = 63 - ((unsigned int)v33 ^ 0x3F);
            }
          }
          v34 = sub_AD64C0(*(_QWORD *)(*(_QWORD *)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF)) + 8LL), v32, 0);
          return sub_B504D0(25, *(_QWORD *)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF)), v34, (__int64)v76, 0, 0);
        }
      }
    }
    v77 = 257;
    return sub_B50550(*(_QWORD *)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF)), (__int64)v76, 0, 0);
  }
  v3 = 0;
  switch ( v6 )
  {
    case 333:
    case 339:
    case 360:
    case 369:
    case 372:
      goto LABEL_10;
    case 334:
    case 335:
    case 336:
    case 337:
    case 338:
    case 340:
    case 341:
    case 342:
    case 343:
    case 344:
    case 345:
    case 346:
    case 347:
    case 348:
    case 349:
    case 350:
    case 351:
    case 352:
    case 353:
    case 354:
    case 355:
    case 356:
    case 357:
    case 358:
    case 359:
    case 361:
    case 362:
    case 363:
    case 364:
    case 365:
    case 366:
    case 367:
    case 368:
    case 370:
    case 371:
      return 0;
    default:
      return (__int64)v3;
  }
  return (__int64)v3;
}
