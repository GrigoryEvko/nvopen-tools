// Function: sub_1A1DB70
// Address: 0x1a1db70
//
_QWORD *__fastcall sub_1A1DB70(__int64 a1, __int64 *a2, __int64 a3, unsigned int a4, const __m128i *a5, int a6)
{
  __int64 v6; // r14
  __int64 *v7; // r12
  __int64 v9; // rdx
  _QWORD *result; // rax
  __int64 v11; // r13
  unsigned __int64 v12; // rdx
  __int64 v13; // rax
  int v14; // r8d
  __int64 v15; // r9
  __int64 v16; // rax
  _QWORD *v17; // rdi
  __int64 **v18; // rax
  __int64 v19; // rax
  __int64 v20; // r15
  _QWORD *v21; // rax
  _QWORD *v22; // rbx
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // r14
  _QWORD *v26; // rax
  _QWORD *v27; // r11
  __int64 v28; // r10
  __int64 *v29; // r14
  unsigned int v30; // r15d
  unsigned int v31; // ebx
  __int64 v32; // rax
  int v33; // r8d
  __int64 v34; // r9
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // r15
  _QWORD *v38; // rax
  _QWORD *v39; // r13
  _QWORD *v40; // rbx
  __int64 v41; // [rsp+0h] [rbp-120h]
  _QWORD *v42; // [rsp+0h] [rbp-120h]
  __int64 v43; // [rsp+0h] [rbp-120h]
  __int64 v44; // [rsp+8h] [rbp-118h]
  unsigned int v45; // [rsp+10h] [rbp-110h]
  __int64 v46; // [rsp+10h] [rbp-110h]
  __int64 v47; // [rsp+20h] [rbp-100h]
  _QWORD *v49; // [rsp+38h] [rbp-E8h]
  __m128i v50; // [rsp+40h] [rbp-E0h] BYREF
  char v51; // [rsp+50h] [rbp-D0h]
  char v52; // [rsp+51h] [rbp-CFh]
  __m128i v53; // [rsp+60h] [rbp-C0h] BYREF
  char v54; // [rsp+70h] [rbp-B0h]
  char v55; // [rsp+71h] [rbp-AFh]
  __m128i v56; // [rsp+80h] [rbp-A0h] BYREF
  __int16 v57; // [rsp+90h] [rbp-90h]
  __int64 *v58; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v59; // [rsp+A8h] [rbp-78h]
  _WORD v60[56]; // [rsp+B0h] [rbp-70h] BYREF

  v6 = a3;
  v7 = (__int64 *)a1;
  if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
  {
    v9 = *(_QWORD *)(*(_QWORD *)a3 + 32LL);
    result = (_QWORD *)v6;
    v11 = *a2;
    if ( v9 != *(_QWORD *)(*a2 + 32) )
    {
      v45 = a4 + v9;
      v58 = (__int64 *)v60;
      v59 = 0x800000000LL;
      v12 = *(_QWORD *)(v11 + 32);
      if ( v12 > 8 )
      {
        sub_16CD150((__int64)&v58, v60, v12, 8, (int)a5, a6);
        v12 = *(_QWORD *)(v11 + 32);
      }
      if ( v12 )
      {
        v47 = v6;
        LODWORD(v6) = 0;
        while ( 1 )
        {
          v17 = (_QWORD *)v7[3];
          if ( a4 <= (unsigned int)v6 && v45 > (unsigned int)v6 )
          {
            v13 = sub_1643350(v17);
            v15 = sub_159C470(v13, (unsigned int)v6 - a4, 0);
            v16 = (unsigned int)v59;
            if ( (unsigned int)v59 >= HIDWORD(v59) )
              goto LABEL_12;
          }
          else
          {
            v18 = (__int64 **)sub_1643350(v17);
            v15 = sub_1599EF0(v18);
            v16 = (unsigned int)v59;
            if ( (unsigned int)v59 >= HIDWORD(v59) )
            {
LABEL_12:
              v44 = v15;
              sub_16CD150((__int64)&v58, v60, 0, 8, v14, v15);
              v16 = (unsigned int)v59;
              v15 = v44;
            }
          }
          v58[v16] = v15;
          LODWORD(v59) = v59 + 1;
          v6 = (unsigned int)(v6 + 1);
          if ( v6 == *(_QWORD *)(v11 + 32) )
          {
            v6 = v47;
            break;
          }
        }
      }
      v52 = 1;
      v50.m128i_i64[0] = (__int64)".expand";
      v51 = 3;
      sub_14EC200(&v53, a5, &v50);
      v23 = sub_15A01B0(v58, (unsigned int)v59);
      v24 = sub_1599EF0(*(__int64 ***)v6);
      if ( *(_BYTE *)(v6 + 16) > 0x10u || *(_BYTE *)(v24 + 16) > 0x10u || *(_BYTE *)(v23 + 16) > 0x10u )
      {
        v41 = v24;
        v57 = 257;
        v26 = sub_1648A60(56, 3u);
        v27 = v26;
        if ( v26 )
        {
          v28 = v41;
          v42 = v26;
          sub_15FA660((__int64)v26, (_QWORD *)v6, v28, (_QWORD *)v23, (__int64)&v56, 0);
          v27 = v42;
        }
        v25 = (__int64)sub_1A1C7B0(v7, v27, &v53);
      }
      else
      {
        v25 = sub_15A3950(v6, v24, (_BYTE *)v23, 0);
      }
      LODWORD(v59) = 0;
      if ( *(_QWORD *)(v11 + 32) )
      {
        v43 = v25;
        v29 = v7;
        LODWORD(v7) = 0;
        v30 = a4;
        v31 = v45;
        do
        {
          v32 = sub_1643320((_QWORD *)v29[3]);
          v34 = sub_159C470(v32, (v30 <= (unsigned int)v7) & (unsigned __int8)(v31 > (unsigned int)v7), 0);
          v35 = (unsigned int)v59;
          if ( (unsigned int)v59 >= HIDWORD(v59) )
          {
            v46 = v34;
            sub_16CD150((__int64)&v58, v60, 0, 8, v33, v34);
            v35 = (unsigned int)v59;
            v34 = v46;
          }
          v58[v35] = v34;
          LODWORD(v59) = v59 + 1;
          v7 = (__int64 *)(unsigned int)((_DWORD)v7 + 1);
        }
        while ( v7 != *(__int64 **)(v11 + 32) );
        v7 = v29;
        v25 = v43;
      }
      v52 = 1;
      v50.m128i_i64[0] = (__int64)"blend";
      v51 = 3;
      sub_14EC200(&v53, a5, &v50);
      v36 = sub_15A01B0(v58, (unsigned int)v59);
      v37 = v36;
      if ( *(_BYTE *)(v36 + 16) > 0x10u || *(_BYTE *)(v25 + 16) > 0x10u || *((_BYTE *)a2 + 16) > 0x10u )
      {
        v57 = 257;
        v38 = sub_1648A60(56, 3u);
        v39 = v38;
        if ( v38 )
        {
          v40 = v38 - 9;
          sub_15F1EA0((__int64)v38, *(_QWORD *)v25, 55, (__int64)(v38 - 9), 3, 0);
          sub_1593B40(v40, v37);
          sub_1593B40(v39 - 6, v25);
          sub_1593B40(v39 - 3, (__int64)a2);
          sub_164B780((__int64)v39, v56.m128i_i64);
        }
        result = sub_1A1C7B0(v7, v39, &v53);
      }
      else
      {
        result = (_QWORD *)sub_15A2DC0(v36, (__int64 *)v25, (__int64)a2, 0);
      }
      if ( v58 != (__int64 *)v60 )
      {
        v49 = result;
        _libc_free((unsigned __int64)v58);
        return v49;
      }
    }
  }
  else
  {
    v55 = 1;
    v53.m128i_i64[0] = (__int64)".insert";
    v54 = 3;
    sub_14EC200(&v56, a5, &v53);
    v19 = sub_1643350(*(_QWORD **)(a1 + 24));
    v20 = sub_159C470(v19, a4, 0);
    if ( *((_BYTE *)a2 + 16) <= 0x10u && *(_BYTE *)(v6 + 16) <= 0x10u && *(_BYTE *)(v20 + 16) <= 0x10u )
    {
      return (_QWORD *)sub_15A3890(a2, v6, v20, 0);
    }
    else
    {
      v60[0] = 257;
      v21 = sub_1648A60(56, 3u);
      v22 = v21;
      if ( v21 )
        sub_15FA480((__int64)v21, a2, v6, v20, (__int64)&v58, 0);
      return sub_1A1C7B0((__int64 *)a1, v22, &v56);
    }
  }
  return result;
}
