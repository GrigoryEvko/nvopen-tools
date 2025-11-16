// Function: sub_3050510
// Address: 0x3050510
//
__int64 __fastcall sub_3050510(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r12d
  unsigned int v7; // r13d
  __int64 v8; // r15
  __int64 result; // rax
  __int64 v11; // rbx
  __int64 v12; // rsi
  __int64 v13; // r13
  __int64 v14; // rax
  unsigned __int16 v15; // bx
  __int64 v16; // rax
  int v17; // eax
  __int64 v18; // r13
  __int128 *v19; // rbx
  __int128 *v20; // r15
  __int128 v21; // rax
  int v22; // r9d
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // r8
  __int64 *v27; // rdx
  __int64 v28; // rax
  _BYTE *v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // r8
  __int64 *v34; // rdx
  __int64 v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rsi
  __int64 v38; // r12
  __int128 *v39; // rax
  __int64 v40; // rsi
  __int128 v41; // rcx
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rsi
  __int128 v46; // [rsp-10h] [rbp-120h]
  __int128 v47; // [rsp-10h] [rbp-120h]
  __int64 v48; // [rsp+10h] [rbp-100h]
  __int64 v49; // [rsp+20h] [rbp-F0h]
  __int64 v50; // [rsp+20h] [rbp-F0h]
  __int64 v51; // [rsp+28h] [rbp-E8h]
  __int64 v52; // [rsp+28h] [rbp-E8h]
  __int64 v53; // [rsp+30h] [rbp-E0h]
  __int64 v54; // [rsp+38h] [rbp-D8h]
  __int64 v55; // [rsp+40h] [rbp-D0h]
  __int64 v56; // [rsp+50h] [rbp-C0h] BYREF
  int v57; // [rsp+58h] [rbp-B8h]
  _BYTE *v58; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v59; // [rsp+68h] [rbp-A8h]
  _BYTE v60[48]; // [rsp+70h] [rbp-A0h] BYREF
  _BYTE *v61; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v62; // [rsp+A8h] [rbp-68h]
  _BYTE v63[96]; // [rsp+B0h] [rbp-60h] BYREF

  v6 = *(_DWORD *)(a2 + 24);
  v7 = a3;
  v8 = a2;
  switch ( v6 )
  {
    case 13:
      return sub_3038030((__int64)a1, a2, a3, a4);
    case 22:
    case 23:
      return 0;
    case 46:
      return sub_3050440((__int64)a1, a2, a3, a4);
    case 47:
      return sub_304E6C0((__int64)a1, a2, a3, a4, a5, a6);
    case 48:
      return sub_30462A0((__int64)a1, a2, a3, a4, a5, a6);
    case 49:
      return sub_303E210((__int64)a1, a2, a3, a4);
    case 53:
      return sub_30477B0((__int64)a1, a2, a3, a4);
    case 54:
      return sub_30476E0((__int64)a1, a2, a3, a4, a5, a6);
    case 56:
    case 57:
    case 58:
    case 61:
    case 62:
    case 180:
    case 181:
    case 182:
    case 183:
    case 189:
    case 190:
      v11 = a2;
      v12 = *(_QWORD *)(a2 + 80);
      v56 = v12;
      if ( v12 )
        sub_B96E90((__int64)&v56, v12, 1);
      v13 = 16LL * v7;
      v57 = *(_DWORD *)(v8 + 72);
      v14 = *(_QWORD *)(v8 + 48) + v13;
      v48 = v13;
      if ( *(_WORD *)v14 == 47 )
      {
        v58 = v60;
        v59 = 0x300000000LL;
        v15 = *(_WORD *)v14;
        v16 = *(_QWORD *)(v14 + 8);
        LOWORD(v61) = v15;
        v62 = v16;
        if ( v15 )
        {
          if ( (unsigned __int16)(v15 - 176) <= 0x34u )
          {
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
            sub_CA17B0(
              "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se MVT::getVectorElementCount() instead");
          }
          v17 = word_4456340[v15 - 1];
        }
        else
        {
          if ( sub_3007100((__int64)&v61) )
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
          v17 = sub_3007130((__int64)&v61, v12);
        }
        if ( v17 > 0 )
        {
          v54 = v8;
          v18 = 0;
          v53 = (unsigned int)v17;
          do
          {
            v61 = v63;
            v62 = 0x300000000LL;
            v19 = *(__int128 **)(v54 + 40);
            v20 = (__int128 *)((char *)v19 + 40 * *(unsigned int *)(v54 + 64));
            if ( v19 == v20 )
            {
              v29 = v63;
              v28 = 0;
            }
            else
            {
              do
              {
                *(_QWORD *)&v21 = sub_3400D50(a4, v18, &v56, 0);
                v23 = sub_3406EB0(a4, 158, (unsigned int)&v56, 6, 0, v22, *v19, v21);
                a6 = v24;
                v25 = (unsigned int)v62;
                v26 = v23;
                if ( (unsigned __int64)(unsigned int)v62 + 1 > HIDWORD(v62) )
                {
                  v49 = v23;
                  v51 = a6;
                  sub_C8D5F0((__int64)&v61, v63, (unsigned int)v62 + 1LL, 0x10u, v23, a6);
                  v25 = (unsigned int)v62;
                  v26 = v49;
                  a6 = v51;
                }
                v27 = (__int64 *)&v61[16 * v25];
                v19 = (__int128 *)((char *)v19 + 40);
                *v27 = v26;
                v27[1] = a6;
                v28 = (unsigned int)(v62 + 1);
                LODWORD(v62) = v62 + 1;
              }
              while ( v20 != v19 );
              v29 = v61;
            }
            *((_QWORD *)&v46 + 1) = v28;
            *(_QWORD *)&v46 = v29;
            v30 = sub_33FC220(a4, *(_DWORD *)(v54 + 24), (unsigned int)&v56, 6, 0, a6, v46);
            a6 = v31;
            v32 = (unsigned int)v59;
            v33 = v30;
            if ( (unsigned __int64)(unsigned int)v59 + 1 > HIDWORD(v59) )
            {
              v50 = v30;
              v52 = a6;
              sub_C8D5F0((__int64)&v58, v60, (unsigned int)v59 + 1LL, 0x10u, v30, a6);
              v32 = (unsigned int)v59;
              v33 = v50;
              a6 = v52;
            }
            v34 = (__int64 *)&v58[16 * v32];
            *v34 = v33;
            v34[1] = a6;
            LODWORD(v59) = v59 + 1;
            if ( v61 != v63 )
              _libc_free((unsigned __int64)v61);
            ++v18;
          }
          while ( v18 != v53 );
          v8 = v54;
        }
        *((_QWORD *)&v47 + 1) = (unsigned int)v59;
        *(_QWORD *)&v47 = v58;
        v11 = sub_33FC220(
                a4,
                156,
                (unsigned int)&v56,
                *(unsigned __int16 *)(*(_QWORD *)(v8 + 48) + v48),
                *(_QWORD *)(*(_QWORD *)(v8 + 48) + v48 + 8),
                a6,
                v47);
        if ( v58 != v60 )
          _libc_free((unsigned __int64)v58);
      }
      if ( v56 )
        sub_B91220((__int64)&v56, v56);
      return v11;
    case 96:
    case 97:
    case 98:
      return sub_303B880((__int64)a1, a2, a3, a4);
    case 152:
      return sub_303AC30((__int64)a1, a2, a3, a4, a5, a6);
    case 156:
      return sub_3039000((__int64)a1, a2, a3, a4);
    case 157:
      return sub_3039F20((__int64)a1, a2, a3, a4);
    case 158:
      return sub_3039B70((__int64)a1, a2, a3, a4);
    case 159:
      return sub_30389E0((__int64)a1, a2, a3, a4, a5, a6);
    case 161:
      return a2;
    case 165:
      return sub_303A100((__int64)a1, a2, a3, a4);
    case 193:
    case 194:
      v35 = *(_QWORD *)(a2 + 80);
      v61 = (_BYTE *)v35;
      if ( v35 )
        sub_B96E90((__int64)&v61, v35, 1);
      LODWORD(v62) = *(_DWORD *)(v8 + 72);
      v36 = sub_302F1D0(
              **(_QWORD **)(v8 + 40),
              *(_QWORD *)(*(_QWORD *)(v8 + 40) + 8LL),
              **(_QWORD **)(v8 + 40),
              *(_QWORD *)(*(_QWORD *)(v8 + 40) + 8LL),
              *(_QWORD *)(*(_QWORD *)(v8 + 40) + 40LL),
              (int)&v61,
              (unsigned int)(v6 != 193) + 195,
              a4);
      goto LABEL_34;
    case 195:
    case 196:
      v45 = *(_QWORD *)(a2 + 80);
      v61 = (_BYTE *)v45;
      if ( v45 )
        sub_B96E90((__int64)&v61, v45, 1);
      LODWORD(v62) = *(_DWORD *)(v8 + 72);
      v36 = sub_302F1D0(
              **(_QWORD **)(v8 + 40),
              *(_QWORD *)(*(_QWORD *)(v8 + 40) + 8LL),
              *(_QWORD *)(*(_QWORD *)(v8 + 40) + 40LL),
              *(_QWORD *)(*(_QWORD *)(v8 + 40) + 48LL),
              *(_QWORD *)(*(_QWORD *)(v8 + 40) + 80LL),
              (int)&v61,
              v6,
              a4);
LABEL_34:
      v37 = (__int64)v61;
      v38 = v36;
      if ( v61 )
        goto LABEL_35;
      goto LABEL_36;
    case 199:
    case 200:
      v39 = *(__int128 **)(a2 + 40);
      v40 = *(_QWORD *)(a2 + 80);
      v41 = *v39;
      v61 = (_BYTE *)v40;
      if ( v40 )
      {
        v55 = v41;
        sub_B96E90((__int64)&v61, v40, 1);
        v6 = *(_DWORD *)(v8 + 24);
        *(_QWORD *)&v41 = v55;
      }
      LODWORD(v62) = *(_DWORD *)(v8 + 72);
      v42 = sub_33FAF80(a4, v6, (unsigned int)&v61, 7, 0, a6, v41);
      v44 = sub_33FA050(a4, 214, (unsigned int)&v61, 8, 0, 16, v42, v43);
      v37 = (__int64)v61;
      v38 = v44;
      if ( !v61 )
        goto LABEL_36;
LABEL_35:
      sub_B91220((__int64)&v61, v37);
LABEL_36:
      result = v38;
      break;
    case 205:
      result = sub_303D290((__int64)a1, a2, a3, a4);
      break;
    case 210:
      result = sub_303A7B0((__int64)a1, a2, a3, a4);
      break;
    case 211:
    case 212:
      result = sub_303A310((__int64)a1, a2, a3, a4);
      break;
    case 220:
    case 221:
      result = sub_303BA80((__int64)a1, a2, a3, a4);
      break;
    case 226:
    case 227:
      result = sub_303BB80((__int64)a1, a2, a3, a4, a5, a6);
      break;
    case 230:
      result = sub_303BC70((__int64)a1, a2, a3, a4);
      break;
    case 233:
      result = sub_303BFD0((__int64)a1, a2, a3, a4, a5, a6);
      break;
    case 234:
      result = sub_3038D40((__int64)a1, a2, a3, a4);
      break;
    case 235:
      result = sub_303C680((__int64)a1, a2, a3, a4);
      break;
    case 272:
      result = sub_303B850((__int64)a1, a2, a3, a4, a5, a6);
      break;
    case 298:
      result = sub_303D530((__int64)a1, a2, a3, a4);
      break;
    case 299:
      result = sub_303E0D0((__int64)a1, a2, a3, a4, a5, a6);
      break;
    case 300:
      result = sub_30381D0((__int64)a1, a2, a3, a4);
      break;
    case 303:
      result = sub_303C3F0((__int64)a1, a2, a3, a4);
      break;
    case 313:
      result = sub_3038700(a1, a2, a3, a4);
      break;
    case 314:
      result = sub_3038500(a1, a2, a3, a4);
      break;
    case 317:
      result = sub_303C710((__int64)a1, a2, a3, a4);
      break;
    case 320:
      result = sub_3045D90((__int64)a1, a2, a3, a4);
      break;
    default:
      BUG();
  }
  return result;
}
