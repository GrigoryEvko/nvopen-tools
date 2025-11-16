// Function: sub_2DB8E00
// Address: 0x2db8e00
//
__int64 __fastcall sub_2DB8E00(__int64 a1, __int64 *a2)
{
  void *v3; // rdx
  __int64 v4; // rbx
  _WORD *v5; // rdx
  int v6; // ebx
  __int64 v7; // r15
  __int64 v8; // rsi
  _BYTE *v9; // rdi
  __int64 v10; // rdx
  __m128i *v11; // rdx
  __m128i si128; // xmm0
  _DWORD *v13; // rdx
  _BYTE *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r12
  _WORD *v18; // rdx
  _WORD *v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdi
  _BYTE *v22; // rax
  __int64 *v23; // rbx
  __int64 v24; // r14
  __int64 v25; // rdx
  __m128i *v26; // rdx
  __m128i v27; // xmm0
  _WORD *v28; // rdx
  __int64 v30; // rax
  __int64 v31; // [rsp+0h] [rbp-120h]
  __int64 *v33; // [rsp+18h] [rbp-108h]
  __int64 v34; // [rsp+20h] [rbp-100h]
  __int64 v35; // [rsp+28h] [rbp-F8h]
  _BYTE v36[16]; // [rsp+30h] [rbp-F0h] BYREF
  void (__fastcall *v37)(_BYTE *, _BYTE *, __int64); // [rsp+40h] [rbp-E0h]
  void (__fastcall *v38)(_BYTE *, __int64); // [rsp+48h] [rbp-D8h]
  _BYTE v39[16]; // [rsp+50h] [rbp-D0h] BYREF
  void (__fastcall *v40)(_BYTE *, _BYTE *, __int64); // [rsp+60h] [rbp-C0h]
  void (__fastcall *v41)(_BYTE *, __int64); // [rsp+68h] [rbp-B8h]
  _BYTE v42[16]; // [rsp+70h] [rbp-B0h] BYREF
  void (__fastcall *v43)(_BYTE *, _BYTE *, __int64); // [rsp+80h] [rbp-A0h]
  void (__fastcall *v44)(_BYTE *, __int64); // [rsp+88h] [rbp-98h]
  _BYTE v45[16]; // [rsp+90h] [rbp-90h] BYREF
  void (__fastcall *v46)(_BYTE *, _BYTE *, __int64); // [rsp+A0h] [rbp-80h]
  void (__fastcall *v47)(_BYTE *, __int64); // [rsp+A8h] [rbp-78h]
  _BYTE v48[16]; // [rsp+B0h] [rbp-70h] BYREF
  void (__fastcall *v49)(_BYTE *, _BYTE *, __int64); // [rsp+C0h] [rbp-60h]
  void (__fastcall *v50)(_BYTE *, __int64); // [rsp+C8h] [rbp-58h]
  _BYTE v51[16]; // [rsp+D0h] [rbp-50h] BYREF
  void (__fastcall *v52)(_BYTE *, _BYTE *, __int64); // [rsp+E0h] [rbp-40h]
  void (__fastcall *v53)(_BYTE *, __int64); // [rsp+E8h] [rbp-38h]

  v3 = *(void **)(a1 + 32);
  v4 = *a2;
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v3 <= 9u )
  {
    sub_CB6200(a1, "digraph {\n", 0xAu);
    v5 = *(_WORD **)(a1 + 32);
  }
  else
  {
    qmemcpy(v3, "digraph {\n", 10);
    v5 = (_WORD *)(*(_QWORD *)(a1 + 32) + 10LL);
    *(_QWORD *)(a1 + 32) = v5;
  }
  v31 = v4 + 320;
  v35 = *(_QWORD *)(v4 + 328);
  if ( v35 != v4 + 320 )
  {
    while ( 1 )
    {
      v6 = *(_DWORD *)(v35 + 24);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v5 <= 1u )
      {
        v7 = sub_CB6200(a1, (unsigned __int8 *)"\t\"", 2u);
      }
      else
      {
        v7 = a1;
        *v5 = 8713;
        *(_QWORD *)(a1 + 32) += 2LL;
      }
      v8 = v35;
      v9 = v36;
      sub_2E31000(v36, v35);
      if ( !v37 )
        goto LABEL_67;
      v38(v36, v7);
      v11 = *(__m128i **)(v7 + 32);
      if ( *(_QWORD *)(v7 + 24) - (_QWORD)v11 <= 0x15u )
      {
        v7 = sub_CB6200(v7, "\" [ shape=box, label=\"", 0x16u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_444ED30);
        v11[1].m128i_i32[0] = 1818583649;
        v11[1].m128i_i16[2] = 8765;
        *v11 = si128;
        *(_QWORD *)(v7 + 32) += 22LL;
      }
      v8 = v35;
      v9 = v39;
      sub_2E31000(v39, v35);
      if ( !v40 )
        goto LABEL_67;
      v41(v39, v7);
      v13 = *(_DWORD **)(v7 + 32);
      if ( *(_QWORD *)(v7 + 24) - (_QWORD)v13 <= 3u )
      {
        v7 = sub_CB6200(v7, "\" ]\n", 4u);
        v14 = *(_BYTE **)(v7 + 32);
      }
      else
      {
        *v13 = 173875234;
        v14 = (_BYTE *)(*(_QWORD *)(v7 + 32) + 4LL);
        *(_QWORD *)(v7 + 32) = v14;
      }
      if ( *(_QWORD *)(v7 + 24) <= (unsigned __int64)v14 )
      {
        v7 = sub_CB5D20(v7, 9);
      }
      else
      {
        *(_QWORD *)(v7 + 32) = v14 + 1;
        *v14 = 9;
      }
      v15 = sub_CB59D0(v7, *(unsigned int *)(a2[1] + 4LL * (unsigned int)(2 * v6)));
      v16 = *(_QWORD *)(v15 + 32);
      v17 = v15;
      if ( (unsigned __int64)(*(_QWORD *)(v15 + 24) - v16) <= 4 )
      {
        v17 = sub_CB6200(v15, (unsigned __int8 *)" -> \"", 5u);
      }
      else
      {
        *(_DWORD *)v16 = 540945696;
        *(_BYTE *)(v16 + 4) = 34;
        *(_QWORD *)(v15 + 32) += 5LL;
      }
      v8 = v35;
      v9 = v42;
      sub_2E31000(v42, v35);
      if ( !v43 )
        goto LABEL_67;
      v44(v42, v17);
      v18 = *(_WORD **)(v17 + 32);
      if ( *(_QWORD *)(v17 + 24) - (_QWORD)v18 <= 1u )
      {
        v30 = sub_CB6200(v17, (unsigned __int8 *)"\"\n", 2u);
        v19 = *(_WORD **)(v30 + 32);
        v17 = v30;
      }
      else
      {
        *v18 = 2594;
        v19 = (_WORD *)(*(_QWORD *)(v17 + 32) + 2LL);
        *(_QWORD *)(v17 + 32) = v19;
      }
      if ( *(_QWORD *)(v17 + 24) - (_QWORD)v19 <= 1u )
      {
        v17 = sub_CB6200(v17, (unsigned __int8 *)"\t\"", 2u);
      }
      else
      {
        *v19 = 8713;
        *(_QWORD *)(v17 + 32) += 2LL;
      }
      v8 = v35;
      v9 = v45;
      sub_2E31000(v45, v35);
      if ( !v46 )
        goto LABEL_67;
      v47(v45, v17);
      v20 = *(_QWORD *)(v17 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v17 + 24) - v20) <= 4 )
      {
        v17 = sub_CB6200(v17, "\" -> ", 5u);
      }
      else
      {
        *(_DWORD *)v20 = 1043144738;
        *(_BYTE *)(v20 + 4) = 32;
        *(_QWORD *)(v17 + 32) += 5LL;
      }
      v21 = sub_CB59D0(v17, *(unsigned int *)(a2[1] + 4LL * (unsigned int)(2 * v6 + 1)));
      v22 = *(_BYTE **)(v21 + 32);
      if ( (unsigned __int64)v22 >= *(_QWORD *)(v21 + 24) )
      {
        sub_CB5D20(v21, 10);
      }
      else
      {
        *(_QWORD *)(v21 + 32) = v22 + 1;
        *v22 = 10;
      }
      if ( v46 )
        v46(v45, v45, 3);
      if ( v43 )
        v43(v42, v42, 3);
      if ( v40 )
        v40(v39, v39, 3);
      if ( v37 )
        v37(v36, v36, 3);
      v23 = *(__int64 **)(v35 + 112);
      v33 = &v23[*(unsigned int *)(v35 + 120)];
      if ( v23 != v33 )
        break;
LABEL_52:
      v5 = *(_WORD **)(a1 + 32);
      v35 = *(_QWORD *)(v35 + 8);
      if ( v31 == v35 )
        goto LABEL_53;
    }
    while ( 1 )
    {
      v28 = *(_WORD **)(a1 + 32);
      v34 = *v23;
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v28 > 1u )
      {
        v24 = a1;
        *v28 = 8713;
        *(_QWORD *)(a1 + 32) += 2LL;
      }
      else
      {
        v24 = sub_CB6200(a1, (unsigned __int8 *)"\t\"", 2u);
      }
      v8 = v35;
      v9 = v48;
      sub_2E31000(v48, v35);
      if ( !v49 )
        break;
      v50(v48, v24);
      v25 = *(_QWORD *)(v24 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v24 + 24) - v25) <= 5 )
      {
        v24 = sub_CB6200(v24, "\" -> \"", 6u);
      }
      else
      {
        *(_DWORD *)v25 = 1043144738;
        *(_WORD *)(v25 + 4) = 8736;
        *(_QWORD *)(v24 + 32) += 6LL;
      }
      v8 = v34;
      v9 = v51;
      sub_2E31000(v51, v34);
      if ( !v52 )
        break;
      v53(v51, v24);
      v26 = *(__m128i **)(v24 + 32);
      if ( *(_QWORD *)(v24 + 24) - (_QWORD)v26 <= 0x15u )
      {
        sub_CB6200(v24, "\" [ color=lightgray ]\n", 0x16u);
      }
      else
      {
        v27 = _mm_load_si128((const __m128i *)&xmmword_430B040);
        v26[1].m128i_i32[0] = 544825714;
        v26[1].m128i_i16[2] = 2653;
        *v26 = v27;
        *(_QWORD *)(v24 + 32) += 22LL;
      }
      if ( v52 )
        v52(v51, v51, 3);
      if ( v49 )
        v49(v48, v48, 3);
      if ( v33 == ++v23 )
        goto LABEL_52;
    }
LABEL_67:
    sub_4263D6(v9, v8, v10);
  }
LABEL_53:
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v5 <= 1u )
  {
    sub_CB6200(a1, "}\n", 2u);
  }
  else
  {
    *v5 = 2685;
    *(_QWORD *)(a1 + 32) += 2LL;
  }
  return a1;
}
