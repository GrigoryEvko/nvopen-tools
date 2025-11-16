// Function: sub_12A6630
// Address: 0x12a6630
//
__int64 __fastcall sub_12A6630(_QWORD **a1, __int64 a2, __int64 i, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdi
  unsigned __int8 v9; // al
  __int64 result; // rax
  unsigned __int8 v11; // al
  __int64 v12; // rax
  unsigned __int64 *v13; // r14
  __int64 v14; // r15
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  _QWORD *v19; // rdi
  _QWORD *v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rbx
  __int64 v23; // rcx
  __int64 v24; // rcx
  __int64 v25; // r8
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // r10
  unsigned int v28; // r11d
  char v29; // bl
  __int64 *v30; // r15
  unsigned int v31; // r9d
  unsigned __int64 v32; // r8
  unsigned int v33; // r11d
  unsigned __int64 v34; // r10
  unsigned __int64 v35; // rsi
  __int64 *v36; // rdx
  __int64 v37; // r12
  char *v38; // rax
  _QWORD *v39; // r15
  __int64 v40; // r14
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  __int64 v43; // rbx
  _QWORD *v44; // r15
  _QWORD *v45; // r14
  _QWORD *v46; // r12
  __int64 v47; // rsi
  __int64 v48; // rax
  _QWORD *v49; // rdi
  _QWORD *v50; // rax
  unsigned int v51; // eax
  __int64 v52; // [rsp-10h] [rbp-A0h]
  unsigned int v53; // [rsp+4h] [rbp-8Ch]
  __int64 v54; // [rsp+8h] [rbp-88h]
  unsigned int v55; // [rsp+10h] [rbp-80h]
  unsigned int v56; // [rsp+10h] [rbp-80h]
  unsigned __int64 v57; // [rsp+10h] [rbp-80h]
  _QWORD *v58; // [rsp+18h] [rbp-78h]
  unsigned __int64 v59; // [rsp+18h] [rbp-78h]
  __int64 v60; // [rsp+18h] [rbp-78h]
  unsigned int v61; // [rsp+18h] [rbp-78h]
  __int64 v62; // [rsp+20h] [rbp-70h]
  __int32 v63; // [rsp+28h] [rbp-68h]
  __int64 v64; // [rsp+28h] [rbp-68h]
  _QWORD *v65; // [rsp+28h] [rbp-68h]
  unsigned __int64 v66; // [rsp+28h] [rbp-68h]
  __m128i v67; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v68; // [rsp+40h] [rbp-50h]
  int v69; // [rsp+58h] [rbp-38h]

  while ( 2 )
  {
    if ( !(dword_4D04720 | dword_4D04658) && (*(_WORD *)(a2 + 24) & 0x10FF) != 0x1002 )
    {
      v8 = (__int64)*a1;
      v67.m128i_i64[0] = *(_QWORD *)(a2 + 36);
      if ( v67.m128i_i32[0] )
      {
        sub_1290930(v8, (unsigned int *)&v67);
        sub_127C770(&v67);
      }
    }
    v9 = *(_BYTE *)(a2 + 24);
    if ( v9 == 17 )
    {
      sub_1296570(
        &v67,
        (__int64)*a1,
        *(unsigned int **)(a2 + 56),
        1,
        (__int64)a1[2],
        *((_DWORD *)a1 + 6),
        *((_BYTE *)a1 + 28));
      return v52;
    }
    else if ( v9 > 0x11u )
    {
      if ( v9 != 19 )
        goto LABEL_15;
      return sub_1281220((__int64)&v67, (__int64)*a1, (__int64 *)a2);
    }
    else
    {
      if ( v9 != 1 )
      {
        if ( v9 == 3 )
          return sub_12A6560((__int64)a1, a2, i, a4, a5);
LABEL_15:
        sub_127B550("unexpected expression with aggregate type!", (_DWORD *)(a2 + 36), 1);
      }
      v11 = *(_BYTE *)(a2 + 56);
      if ( v11 <= 0x19u )
      {
        if ( v11 > 2u )
        {
          switch ( v11 )
          {
            case 3u:
            case 6u:
            case 8u:
              return sub_12A6560((__int64)a1, a2, i, a4, a5);
            case 5u:
              if ( a1[2] )
              {
                v12 = *(_QWORD *)a2;
                for ( i = *(unsigned __int8 *)(*(_QWORD *)a2 + 140LL); (_BYTE)i == 12; i = *(unsigned __int8 *)(v12 + 140) )
                  v12 = *(_QWORD *)(v12 + 160);
                if ( (_BYTE)i != 1 )
                  sub_127B550("casting aggregate to non-void type is not supported!", (_DWORD *)(a2 + 36), 1);
              }
              goto LABEL_24;
            case 0x19u:
LABEL_24:
              a2 = *(_QWORD *)(a2 + 72);
              continue;
            default:
              goto LABEL_15;
          }
        }
        goto LABEL_15;
      }
      switch ( v11 )
      {
        case 'I':
          v13 = *(unsigned __int64 **)(a2 + 72);
          v14 = v13[2];
          if ( dword_4D04810
            && ((unsigned int)sub_731770(*(_QWORD *)(a2 + 72), 0, i, a4, a5, a6)
             || (unsigned int)sub_731770(v14, 0, v15, v16, v17, v18)) )
          {
            v19 = *a1;
            v67.m128i_i64[0] = (__int64)"agg.tmp";
            LOWORD(v68) = 259;
            v20 = sub_127FE40(v19, *v13, (__int64)&v67);
            v21 = *v13;
            v22 = (__int64)*a1;
            v58 = v20;
            if ( *(char *)(*v13 + 142) >= 0 && *(_BYTE *)(v21 + 140) == 12 )
              v23 = (unsigned int)sub_8D4AB0(v21);
            else
              v23 = *(unsigned int *)(v21 + 136);
            sub_12A6C40(v22, v14, v58, v23, 0);
            sub_1286D80((__int64)&v67, *a1, (__int64)v13, v24, v25);
            v26 = *v13;
            v27 = v67.m128i_u64[1];
            v28 = v68;
            v29 = v69;
            v63 = v67.m128i_i32[0];
            v30 = *a1;
            if ( *(char *)(*v13 + 142) >= 0 && *(_BYTE *)(v26 + 140) == 12 )
            {
              v53 = v68;
              v54 = v67.m128i_i64[1];
              v57 = *v13;
              v51 = sub_8D4AB0(v26);
              v28 = v53;
              v27 = v54;
              v26 = v57;
              v31 = v51;
            }
            else
            {
              v31 = *(_DWORD *)(v26 + 136);
            }
            v32 = (unsigned __int64)v58;
            v55 = v28;
            v59 = v27;
            result = sub_12A6300(v30, v27, v28, v29 & 1, v32, v31, 0, v26);
            v33 = v55;
            v34 = v59;
          }
          else
          {
            sub_1286D80((__int64)&v67, *a1, (__int64)v13, a4, a5);
            v29 = v69;
            v60 = v67.m128i_i64[1];
            v56 = v68;
            v63 = v67.m128i_i32[0];
            result = sub_12A6C40(*a1, v14, v67.m128i_i64[1], v68, v69 & 1);
            v34 = v60;
            v33 = v56;
          }
          if ( v63 )
            sub_127B550("unexpected aggregate source type!", (_DWORD *)(a2 + 36), 1);
          v35 = (unsigned __int64)a1[2];
          if ( v35 )
            goto LABEL_37;
          v61 = v33;
          v66 = v34;
          if ( (v29 & 1) != 0 )
          {
            v49 = *a1;
            v67.m128i_i64[0] = (__int64)"agg.tmp";
            LOWORD(v68) = 259;
            v50 = sub_127FE40(v49, *(_QWORD *)a2, (__int64)&v67);
            v33 = v61;
            v34 = v66;
            a1[2] = v50;
            v35 = (unsigned __int64)v50;
LABEL_37:
            result = sub_12A6300(*a1, v35, *((_DWORD *)a1 + 6), *((_BYTE *)a1 + 28), v34, v33, v29 & 1, *(_QWORD *)a2);
          }
          break;
        case '[':
          v36 = *(__int64 **)(a2 + 72);
          v37 = v36[2];
          sub_127FF60((__int64)&v67, (__int64)*a1, v36, 0, 0, 0);
          return sub_12A6C40(*a1, v37, a1[2], *((unsigned int *)a1 + 6), *((unsigned __int8 *)a1 + 28));
        case '\\':
        case '^':
        case '_':
          return sub_12A6560((__int64)a1, a2, i, a4, a5);
        case 'g':
          v43 = *(_QWORD *)(*(_QWORD *)(a2 + 72) + 16LL);
          v64 = *(_QWORD *)(a2 + 72);
          v62 = *(_QWORD *)(v43 + 16);
          v44 = (_QWORD *)sub_12A4D50((__int64)*a1, (__int64)"cond.true", 0, 0);
          v45 = (_QWORD *)sub_12A4D50((__int64)*a1, (__int64)"cond.false", 0, 0);
          v46 = (_QWORD *)sub_12A4D50((__int64)*a1, (__int64)"cond.end", 0, 0);
          v47 = v64;
          v65 = *a1;
          v48 = sub_127FEC0((__int64)*a1, v47);
          sub_12A4DB0(v65, v48, (__int64)v44, (__int64)v45, 0);
          sub_1290AF0(*a1, v44, 0);
          sub_12A6630(a1, v43);
          sub_12909B0(*a1, (__int64)v46);
          sub_1290AF0(*a1, v45, 0);
          sub_12A6630(a1, v62);
          sub_12909B0(*a1, (__int64)v46);
          return sub_1290AF0(*a1, v46, 0);
        case 'i':
          return sub_1281200((__int64)&v67);
        case 'p':
          v38 = sub_12A4D00(*a1, *(_QWORD *)(a2 + 72));
          v39 = *a1;
          v40 = (__int64)v38;
          v41 = sub_127A030((*a1)[4] + 8LL, *(_QWORD *)a2, 0);
          result = (__int64)sub_12812E0(v39, v40, v41);
          v42 = (unsigned __int64)a1[2];
          if ( v42 )
            return sub_12A61B0(*a1, result, v42, *((_DWORD *)a1 + 6), *((_BYTE *)a1 + 28));
          return result;
        default:
          goto LABEL_15;
      }
    }
    return result;
  }
}
