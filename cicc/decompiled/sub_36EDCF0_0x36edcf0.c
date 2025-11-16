// Function: sub_36EDCF0
// Address: 0x36edcf0
//
void __fastcall sub_36EDCF0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r8
  __int64 v6; // rax
  int v7; // edx
  unsigned __int64 v8; // rsi
  unsigned __int64 *v9; // r11
  const __m128i *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD *v13; // r13
  __int64 v14; // rdx
  _QWORD *v15; // r12
  unsigned __int64 **v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rsi
  int v19; // eax
  const __m128i *v20; // rbx
  unsigned __int64 v21; // rdi
  unsigned __int64 **v22; // r9
  unsigned __int64 v23; // rdx
  const __m128i *v24; // rax
  __m128i *v25; // rcx
  int v26; // esi
  __int64 v27; // rax
  __m128i v28; // xmm0
  __int64 v29; // rax
  __int64 *v30; // rdi
  __int64 v31; // rax
  int v32; // eax
  __int64 v33; // r9
  int v34; // esi
  __m128i v35; // xmm0
  __m128i v36; // xmm0
  __int64 v37; // r12
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  _BOOL4 v41; // esi
  _BOOL4 v42; // esi
  _BOOL4 v43; // esi
  _BOOL4 v44; // esi
  _BOOL4 v45; // esi
  _BOOL4 v46; // esi
  _BOOL4 v47; // esi
  _BOOL4 v48; // esi
  _BOOL4 v49; // esi
  _BOOL4 v50; // esi
  unsigned int v51; // [rsp+Ch] [rbp-124h]
  __m128i v52; // [rsp+10h] [rbp-120h] BYREF
  __m128i v53; // [rsp+20h] [rbp-110h] BYREF
  const __m128i *v54; // [rsp+30h] [rbp-100h]
  __int64 v55; // [rsp+38h] [rbp-F8h]
  __m128i v56; // [rsp+40h] [rbp-F0h] BYREF
  unsigned __int64 **v57; // [rsp+50h] [rbp-E0h]
  unsigned __int64 *v58; // [rsp+58h] [rbp-D8h]
  __int64 v59; // [rsp+60h] [rbp-D0h] BYREF
  int v60; // [rsp+68h] [rbp-C8h]
  unsigned __int64 *v61; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v62; // [rsp+78h] [rbp-B8h]
  _BYTE v63[176]; // [rsp+80h] [rbp-B0h] BYREF

  v3 = a3;
  v6 = *(unsigned int *)(a2 + 64);
  v7 = *(_DWORD *)(a2 + 64);
  if ( (_DWORD)v3 == 1 )
  {
    v9 = (unsigned __int64 *)((unsigned __int64)(v6 - 8) >> 1);
    v8 = ((v6 - 8) & 0xFFFFFFFFFFFFFFFELL) - 2;
  }
  else
  {
    v8 = v6 - 10;
    v9 = (unsigned __int64 *)(v6 - 12);
    if ( (unsigned int)(v3 - 2) > 1 )
      v9 = (unsigned __int64 *)(v6 - 10);
  }
  v10 = *(const __m128i **)(a2 + 40);
  v11 = *(_QWORD *)(v10->m128i_i64[5 * (unsigned int)(v7 - 1)] + 96);
  if ( *(_DWORD *)(v11 + 32) <= 0x40u )
    v55 = *(_QWORD *)(v11 + 24);
  else
    v55 = **(_QWORD **)(v11 + 24);
  v12 = *(_QWORD *)(v10->m128i_i64[5 * (unsigned int)(v7 - 2)] + 96);
  v13 = *(_QWORD **)(v12 + 24);
  if ( *(_DWORD *)(v12 + 32) > 0x40u )
    v13 = (_QWORD *)*v13;
  v14 = *(_QWORD *)(v10->m128i_i64[5 * (unsigned int)(v7 - 3)] + 96);
  v15 = *(_QWORD **)(v14 + 24);
  if ( *(_DWORD *)(v14 + 32) > 0x40u )
    v15 = (_QWORD *)*v15;
  v16 = (unsigned __int64 **)(v8 + 5);
  v17 = v8 + 3;
  v18 = *(_QWORD *)(a2 + 80);
  v57 = v16;
  v59 = v18;
  if ( v18 )
  {
    v53.m128i_i32[0] = v3;
    v56.m128i_i64[0] = v17;
    v58 = v9;
    sub_B96E90((__int64)&v59, v18, 1);
    v10 = *(const __m128i **)(a2 + 40);
    v3 = v53.m128i_u32[0];
    v17 = v56.m128i_i64[0];
    v9 = v58;
  }
  v19 = *(_DWORD *)(a2 + 72);
  v20 = v10 + 5;
  v21 = 40 * v17;
  v58 = (unsigned __int64 *)v63;
  v22 = &v61;
  v60 = v19;
  v61 = (unsigned __int64 *)v63;
  v62 = 0x800000000LL;
  v23 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v17) >> 3);
  v24 = (const __m128i *)((char *)v20 + 40 * v17);
  v25 = (__m128i *)v63;
  v26 = 0;
  if ( v21 > 0x140 )
  {
    v51 = v3;
    v54 = v24;
    v52.m128i_i64[0] = (__int64)v9;
    v53.m128i_i64[0] = v23;
    v56.m128i_i64[0] = (__int64)&v61;
    sub_C8D5F0((__int64)&v61, v58, v23, 0x10u, v3, (__int64)&v61);
    v26 = v62;
    v3 = v51;
    v24 = v54;
    v9 = (unsigned __int64 *)v52.m128i_i64[0];
    LODWORD(v23) = v53.m128i_i32[0];
    v22 = &v61;
    v25 = (__m128i *)&v61[2 * (unsigned int)v62];
  }
  if ( v20 != v24 )
  {
    do
    {
      if ( v25 )
        *v25 = _mm_loadu_si128(v20);
      v20 = (const __m128i *)((char *)v20 + 40);
      ++v25;
    }
    while ( v24 != v20 );
    v26 = v62;
  }
  LODWORD(v23) = v26 + v23;
  LODWORD(v62) = v23;
  v27 = (unsigned int)v23;
  if ( v15 != (_QWORD *)1 )
  {
    if ( v13 != (_QWORD *)1 )
      goto LABEL_21;
    goto LABEL_62;
  }
  v23 = (unsigned int)v23;
  v35 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL * (unsigned int)v57));
  if ( (unsigned __int64)(unsigned int)v23 + 1 > HIDWORD(v62) )
  {
    LODWORD(v54) = v3;
    v53.m128i_i64[0] = (__int64)v9;
    v56.m128i_i64[0] = (__int64)&v61;
    v52 = v35;
    sub_C8D5F0((__int64)&v61, v58, (unsigned int)v23 + 1LL, 0x10u, v3, (__int64)&v61);
    v23 = (unsigned int)v62;
    v3 = (unsigned int)v54;
    v35 = _mm_load_si128(&v52);
    v9 = (unsigned __int64 *)v53.m128i_i64[0];
    v22 = &v61;
  }
  *(__m128i *)&v61[2 * v23] = v35;
  v27 = (unsigned int)(v62 + 1);
  LODWORD(v62) = v62 + 1;
  if ( v13 == (_QWORD *)1 )
  {
LABEL_62:
    v36 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL * (unsigned int)((_DWORD)v57 + 1)));
    if ( v27 + 1 > (unsigned __int64)HIDWORD(v62) )
    {
      v52.m128i_i32[0] = v3;
      v56.m128i_i64[0] = (__int64)v9;
      v57 = &v61;
      v53 = v36;
      sub_C8D5F0((__int64)&v61, v58, v27 + 1, 0x10u, v3, (__int64)&v61);
      v27 = (unsigned int)v62;
      v3 = v52.m128i_u32[0];
      v36 = _mm_load_si128(&v53);
      v9 = (unsigned __int64 *)v56.m128i_i64[0];
      v22 = v57;
    }
    *(__m128i *)&v61[2 * v27] = v36;
    v27 = (unsigned int)(v62 + 1);
    LODWORD(v62) = v62 + 1;
  }
LABEL_21:
  v28 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  if ( v27 + 1 > (unsigned __int64)HIDWORD(v62) )
  {
    v53.m128i_i32[0] = v3;
    v57 = (unsigned __int64 **)v9;
    v56 = v28;
    sub_C8D5F0((__int64)v22, v58, v27 + 1, 0x10u, v3, (__int64)v22);
    v27 = (unsigned int)v62;
    LODWORD(v3) = v53.m128i_i32[0];
    v28 = _mm_load_si128(&v56);
    v9 = (unsigned __int64 *)v57;
  }
  v56.m128i_i32[0] = v3;
  *(__m128i *)&v61[2 * v27] = v28;
  v29 = *(_QWORD *)(a1 + 64);
  v57 = (unsigned __int64 **)v9;
  v30 = *(__int64 **)(v29 + 40);
  LODWORD(v62) = v62 + 1;
  v31 = sub_2E79000(v30);
  v32 = sub_AE2980(v31, 3u)[1];
  switch ( v56.m128i_i32[0] )
  {
    case 2:
      if ( v57 == (unsigned __int64 **)4 )
      {
        if ( v55 == 1 )
        {
          if ( v15 == (_QWORD *)1 )
          {
            if ( v13 == (_QWORD *)1 )
            {
              v34 = 579;
              if ( v32 != 32 )
                v34 = 555;
            }
            else
            {
              v34 = 578;
              if ( v32 != 32 )
                v34 = 554;
            }
          }
          else if ( v13 == (_QWORD *)1 )
          {
            v34 = 577;
            if ( v32 != 32 )
              v34 = 553;
          }
          else
          {
            v34 = 576;
            if ( v32 != 32 )
              v34 = 552;
          }
        }
        else if ( v15 == (_QWORD *)1 )
        {
          if ( v13 == (_QWORD *)1 )
          {
            v34 = 582;
            if ( v32 != 32 )
              v34 = 558;
          }
          else
          {
            v34 = 581;
            if ( v32 != 32 )
              v34 = 557;
          }
        }
        else if ( v13 == (_QWORD *)1 )
        {
          v34 = 580;
          if ( v32 != 32 )
            v34 = 556;
        }
        else
        {
          v34 = 567;
          if ( v32 != 32 )
            v34 = 543;
        }
      }
      else if ( v57 == (unsigned __int64 **)5 )
      {
        if ( v55 == 1 )
        {
          if ( v15 == (_QWORD *)1 )
          {
            if ( v13 == (_QWORD *)1 )
            {
              v34 = 643;
              if ( v32 != 32 )
                v34 = 619;
            }
            else
            {
              v34 = 642;
              if ( v32 != 32 )
                v34 = 618;
            }
          }
          else if ( v13 == (_QWORD *)1 )
          {
            v34 = 641;
            if ( v32 != 32 )
              v34 = 617;
          }
          else
          {
            v34 = 640;
            if ( v32 != 32 )
              v34 = 616;
          }
        }
        else if ( v15 == (_QWORD *)1 )
        {
          if ( v13 == (_QWORD *)1 )
          {
            v34 = 646;
            if ( v32 != 32 )
              v34 = 622;
          }
          else
          {
            v34 = 645;
            if ( v32 != 32 )
              v34 = 621;
          }
        }
        else if ( v13 == (_QWORD *)1 )
        {
          v34 = 644;
          if ( v32 != 32 )
            v34 = 620;
        }
        else
        {
          v34 = 631;
          if ( v32 != 32 )
            v34 = 607;
        }
      }
      else
      {
        if ( v57 != (unsigned __int64 **)3 )
          goto LABEL_402;
        if ( v55 == 1 )
        {
          if ( v15 == (_QWORD *)1 )
          {
            if ( v13 == (_QWORD *)1 )
            {
              v34 = 515;
              if ( v32 != 32 )
                v34 = 491;
            }
            else
            {
              v34 = 514;
              if ( v32 != 32 )
                v34 = 490;
            }
          }
          else if ( v13 == (_QWORD *)1 )
          {
            v34 = 513;
            if ( v32 != 32 )
              v34 = 489;
          }
          else
          {
            v34 = 512;
            if ( v32 != 32 )
              v34 = 488;
          }
        }
        else if ( v15 == (_QWORD *)1 )
        {
          if ( v13 == (_QWORD *)1 )
          {
            v34 = 518;
            if ( v32 != 32 )
              v34 = 494;
          }
          else
          {
            v34 = 517;
            if ( v32 != 32 )
              v34 = 493;
          }
        }
        else if ( v13 == (_QWORD *)1 )
        {
          v34 = 516;
          if ( v32 != 32 )
            v34 = 492;
        }
        else
        {
          v34 = 503;
          if ( v32 != 32 )
            v34 = 479;
        }
      }
      break;
    case 3:
      if ( v57 == (unsigned __int64 **)4 )
      {
        if ( v55 == 1 )
        {
          if ( v15 == (_QWORD *)1 )
          {
            if ( v13 == (_QWORD *)1 )
            {
              v34 = 572;
              if ( v32 != 32 )
                v34 = 548;
            }
            else
            {
              v34 = 571;
              if ( v32 != 32 )
                v34 = 547;
            }
          }
          else if ( v13 == (_QWORD *)1 )
          {
            v34 = 570;
            if ( v32 != 32 )
              v34 = 546;
          }
          else
          {
            v34 = 569;
            if ( v32 != 32 )
              v34 = 545;
          }
        }
        else if ( v15 == (_QWORD *)1 )
        {
          if ( v13 == (_QWORD *)1 )
          {
            v34 = 575;
            if ( v32 != 32 )
              v34 = 551;
          }
          else
          {
            v34 = 574;
            if ( v32 != 32 )
              v34 = 550;
          }
        }
        else if ( v13 == (_QWORD *)1 )
        {
          v34 = 573;
          if ( v32 != 32 )
            v34 = 549;
        }
        else
        {
          v34 = 568;
          if ( v32 != 32 )
            v34 = 544;
        }
      }
      else if ( v57 == (unsigned __int64 **)5 )
      {
        if ( v55 == 1 )
        {
          if ( v15 == (_QWORD *)1 )
          {
            if ( v13 == (_QWORD *)1 )
            {
              v34 = 636;
              if ( v32 != 32 )
                v34 = 612;
            }
            else
            {
              v34 = 635;
              if ( v32 != 32 )
                v34 = 611;
            }
          }
          else if ( v13 == (_QWORD *)1 )
          {
            v34 = 634;
            if ( v32 != 32 )
              v34 = 610;
          }
          else
          {
            v34 = 633;
            if ( v32 != 32 )
              v34 = 609;
          }
        }
        else if ( v15 == (_QWORD *)1 )
        {
          if ( v13 == (_QWORD *)1 )
          {
            v34 = 639;
            if ( v32 != 32 )
              v34 = 615;
          }
          else
          {
            v34 = 638;
            if ( v32 != 32 )
              v34 = 614;
          }
        }
        else if ( v13 == (_QWORD *)1 )
        {
          v34 = 637;
          if ( v32 != 32 )
            v34 = 613;
        }
        else
        {
          v34 = 632;
          if ( v32 != 32 )
            v34 = 608;
        }
      }
      else
      {
        if ( v57 != (unsigned __int64 **)3 )
          goto LABEL_402;
        if ( v55 == 1 )
        {
          if ( v15 == (_QWORD *)1 )
          {
            if ( v13 == (_QWORD *)1 )
            {
              v34 = 508;
              if ( v32 != 32 )
                v34 = 484;
            }
            else
            {
              v34 = 507;
              if ( v32 != 32 )
                v34 = 483;
            }
          }
          else if ( v13 == (_QWORD *)1 )
          {
            v34 = 506;
            if ( v32 != 32 )
              v34 = 482;
          }
          else
          {
            v34 = 505;
            if ( v32 != 32 )
              v34 = 481;
          }
        }
        else if ( v15 == (_QWORD *)1 )
        {
          if ( v13 == (_QWORD *)1 )
          {
            v34 = 511;
            if ( v32 != 32 )
              v34 = 487;
          }
          else
          {
            v34 = 510;
            if ( v32 != 32 )
              v34 = 486;
          }
        }
        else if ( v13 == (_QWORD *)1 )
        {
          v34 = 509;
          if ( v32 != 32 )
            v34 = 485;
        }
        else
        {
          v34 = 504;
          if ( v32 != 32 )
            v34 = 480;
        }
      }
      break;
    case 1:
      if ( v57 != (unsigned __int64 **)4 )
      {
        if ( v57 == (unsigned __int64 **)5 )
        {
          if ( v55 == 1 )
          {
            if ( v15 == (_QWORD *)1 )
            {
              if ( v13 == (_QWORD *)1 )
              {
                v34 = 627;
                if ( v32 != 32 )
                  v34 = 603;
              }
              else
              {
                v34 = 626;
                if ( v32 != 32 )
                  v34 = 602;
              }
            }
            else if ( v13 == (_QWORD *)1 )
            {
              v34 = 625;
              if ( v32 != 32 )
                v34 = 601;
            }
            else
            {
              v34 = 624;
              if ( v32 != 32 )
                v34 = 600;
            }
          }
          else if ( v15 == (_QWORD *)1 )
          {
            if ( v13 == (_QWORD *)1 )
            {
              v34 = 630;
              if ( v32 != 32 )
                v34 = 606;
            }
            else
            {
              v34 = 629;
              if ( v32 != 32 )
                v34 = 605;
            }
          }
          else if ( v13 == (_QWORD *)1 )
          {
            v34 = 628;
            if ( v32 != 32 )
              v34 = 604;
          }
          else
          {
            v34 = 623;
            if ( v32 != 32 )
              v34 = 599;
          }
          break;
        }
        if ( v57 == (unsigned __int64 **)3 )
        {
          if ( v55 == 1 )
          {
            if ( v15 == (_QWORD *)1 )
            {
              if ( v13 == (_QWORD *)1 )
              {
                v34 = 499;
                if ( v32 != 32 )
                  v34 = 475;
              }
              else
              {
                v34 = 498;
                if ( v32 != 32 )
                  v34 = 474;
              }
            }
            else if ( v13 == (_QWORD *)1 )
            {
              v34 = 497;
              if ( v32 != 32 )
                v34 = 473;
            }
            else
            {
              v34 = 496;
              if ( v32 != 32 )
                v34 = 472;
            }
          }
          else if ( v15 == (_QWORD *)1 )
          {
            if ( v13 == (_QWORD *)1 )
            {
              v34 = 502;
              if ( v32 != 32 )
                v34 = 478;
            }
            else
            {
              v34 = 501;
              if ( v32 != 32 )
                v34 = 477;
            }
          }
          else if ( v13 == (_QWORD *)1 )
          {
            v34 = 500;
            if ( v32 != 32 )
              v34 = 476;
          }
          else
          {
            v34 = 495;
            if ( v32 != 32 )
              v34 = 471;
          }
          break;
        }
LABEL_402:
        BUG();
      }
      if ( v55 == 1 )
      {
        if ( v15 == (_QWORD *)1 )
        {
          if ( v13 == (_QWORD *)1 )
          {
            v34 = 563;
            if ( v32 != 32 )
              v34 = 539;
          }
          else
          {
            v34 = 562;
            if ( v32 != 32 )
              v34 = 538;
          }
        }
        else if ( v13 == (_QWORD *)1 )
        {
          v34 = 561;
          if ( v32 != 32 )
            v34 = 537;
        }
        else
        {
          v34 = 560;
          if ( v32 != 32 )
            v34 = 536;
        }
      }
      else if ( v15 == (_QWORD *)1 )
      {
        if ( v13 == (_QWORD *)1 )
        {
          v34 = 566;
          if ( v32 != 32 )
            v34 = 542;
        }
        else
        {
          v34 = 565;
          if ( v32 != 32 )
            v34 = 541;
        }
      }
      else if ( v13 == (_QWORD *)1 )
      {
        v34 = 564;
        if ( v32 != 32 )
          v34 = 540;
      }
      else
      {
        v34 = 559;
        if ( v32 != 32 )
          v34 = 535;
      }
      break;
    default:
      switch ( (unsigned __int64)v57 )
      {
        case 1uLL:
          if ( v55 == 1 )
          {
            if ( v15 == (_QWORD *)1 )
            {
              v45 = v32 != 32;
              if ( v13 == (_QWORD *)1 )
                v34 = 8 * v45 + 443;
              else
                v34 = 8 * v45 + 442;
            }
            else if ( v13 == (_QWORD *)1 )
            {
              v34 = 8 * (v32 != 32) + 441;
            }
            else
            {
              v34 = 8 * (v32 != 32) + 440;
            }
          }
          else if ( v15 == (_QWORD *)1 )
          {
            v46 = v32 != 32;
            if ( v13 == (_QWORD *)1 )
              v34 = 8 * v46 + 446;
            else
              v34 = 8 * v46 + 445;
          }
          else if ( v13 == (_QWORD *)1 )
          {
            v34 = 8 * (v32 != 32) + 444;
          }
          else
          {
            v34 = 8 * (v32 != 32) + 439;
          }
          goto LABEL_70;
        case 2uLL:
          if ( v55 == 1 )
          {
            if ( v15 == (_QWORD *)1 )
            {
              v44 = v32 != 32;
              if ( v13 == (_QWORD *)1 )
                v34 = 8 * v44 + 459;
              else
                v34 = 8 * v44 + 458;
            }
            else if ( v13 == (_QWORD *)1 )
            {
              v34 = 8 * (v32 != 32) + 457;
            }
            else
            {
              v34 = 8 * (v32 != 32) + 456;
            }
          }
          else if ( v15 == (_QWORD *)1 )
          {
            v47 = v32 != 32;
            if ( v13 == (_QWORD *)1 )
              v34 = 8 * v47 + 462;
            else
              v34 = 8 * v47 + 461;
          }
          else if ( v13 == (_QWORD *)1 )
          {
            v34 = 8 * (v32 != 32) + 460;
          }
          else
          {
            v34 = 8 * (v32 != 32) + 455;
          }
          goto LABEL_70;
        case 3uLL:
          if ( v55 == 1 )
          {
            if ( v15 == (_QWORD *)1 )
            {
              v43 = v32 != 32;
              if ( v13 == (_QWORD *)1 )
                v34 = 8 * v43 + 523;
              else
                v34 = 8 * v43 + 522;
            }
            else if ( v13 == (_QWORD *)1 )
            {
              v34 = 8 * (v32 != 32) + 521;
            }
            else
            {
              v34 = 8 * (v32 != 32) + 520;
            }
          }
          else if ( v15 == (_QWORD *)1 )
          {
            v49 = v32 != 32;
            if ( v13 == (_QWORD *)1 )
              v34 = 8 * v49 + 526;
            else
              v34 = 8 * v49 + 525;
          }
          else if ( v13 == (_QWORD *)1 )
          {
            v34 = 8 * (v32 != 32) + 524;
          }
          else
          {
            v34 = 8 * (v32 != 32) + 519;
          }
          goto LABEL_70;
        case 4uLL:
          if ( v55 == 1 )
          {
            if ( v15 == (_QWORD *)1 )
            {
              v42 = v32 != 32;
              if ( v13 == (_QWORD *)1 )
                v34 = 8 * v42 + 587;
              else
                v34 = 8 * v42 + 586;
            }
            else if ( v13 == (_QWORD *)1 )
            {
              v34 = 8 * (v32 != 32) + 585;
            }
            else
            {
              v34 = 8 * (v32 != 32) + 584;
            }
          }
          else if ( v15 == (_QWORD *)1 )
          {
            v48 = v32 != 32;
            if ( v13 == (_QWORD *)1 )
              v34 = 8 * v48 + 590;
            else
              v34 = 8 * v48 + 589;
          }
          else if ( v13 == (_QWORD *)1 )
          {
            v34 = 8 * (v32 != 32) + 588;
          }
          else
          {
            v34 = 8 * (v32 != 32) + 583;
          }
          goto LABEL_70;
        case 5uLL:
          if ( v55 == 1 )
          {
            if ( v15 == (_QWORD *)1 )
            {
              v41 = v32 != 32;
              if ( v13 == (_QWORD *)1 )
                v34 = 8 * v41 + 651;
              else
                v34 = 8 * v41 + 650;
            }
            else if ( v13 == (_QWORD *)1 )
            {
              v34 = 8 * (v32 != 32) + 649;
            }
            else
            {
              v34 = 8 * (v32 != 32) + 648;
            }
          }
          else if ( v15 == (_QWORD *)1 )
          {
            v50 = v32 != 32;
            if ( v13 == (_QWORD *)1 )
              v34 = 8 * v50 + 654;
            else
              v34 = 8 * v50 + 653;
          }
          else if ( v13 == (_QWORD *)1 )
          {
            v34 = 8 * (v32 != 32) + 652;
          }
          else
          {
            v34 = 8 * (v32 != 32) + 647;
          }
          goto LABEL_70;
        default:
          goto LABEL_402;
      }
  }
LABEL_70:
  v37 = sub_33E66D0(
          *(_QWORD **)(a1 + 64),
          v34,
          (__int64)&v59,
          *(_QWORD *)(a2 + 48),
          *(unsigned int *)(a2 + 68),
          v33,
          v61,
          (unsigned int)v62);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v37, v38, v39, v40);
  sub_3421DB0(v37);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v61 != v58 )
    _libc_free((unsigned __int64)v61);
  if ( v59 )
    sub_B91220((__int64)&v59, v59);
}
