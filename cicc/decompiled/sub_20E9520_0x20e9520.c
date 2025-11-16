// Function: sub_20E9520
// Address: 0x20e9520
//
__int64 __fastcall sub_20E9520(__int64 a1, __int64 a2)
{
  void *v3; // rdx
  __int64 v4; // rbx
  _WORD *v5; // rdx
  int v6; // ebx
  __int64 v7; // r12
  __int64 v8; // rsi
  _QWORD *v9; // rdi
  __int64 v10; // rdx
  __m128i *v11; // rdx
  _BYTE *v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rbx
  _WORD *v17; // rdx
  _WORD *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdi
  _BYTE *v22; // rax
  __int64 *v23; // rbx
  __int64 v24; // r14
  __int64 v25; // rdx
  __m128i *v26; // rdx
  __m128i si128; // xmm0
  _WORD *v28; // rdx
  __int64 v30; // [rsp+8h] [rbp-F8h]
  __int64 *v32; // [rsp+20h] [rbp-E0h]
  __int64 v33; // [rsp+28h] [rbp-D8h]
  _QWORD v34[2]; // [rsp+30h] [rbp-D0h] BYREF
  void (__fastcall *v35)(_QWORD *, _QWORD *, __int64); // [rsp+40h] [rbp-C0h]
  void (__fastcall *v36)(_QWORD *, __int64); // [rsp+48h] [rbp-B8h]
  _QWORD v37[2]; // [rsp+50h] [rbp-B0h] BYREF
  void (__fastcall *v38)(_QWORD *, _QWORD *, __int64); // [rsp+60h] [rbp-A0h]
  void (__fastcall *v39)(_QWORD *, __int64); // [rsp+68h] [rbp-98h]
  _QWORD v40[2]; // [rsp+70h] [rbp-90h] BYREF
  void (__fastcall *v41)(_QWORD *, _QWORD *, __int64); // [rsp+80h] [rbp-80h]
  void (__fastcall *v42)(_QWORD *, __int64); // [rsp+88h] [rbp-78h]
  _QWORD v43[2]; // [rsp+90h] [rbp-70h] BYREF
  void (__fastcall *v44)(_QWORD *, _QWORD *, __int64); // [rsp+A0h] [rbp-60h]
  void (__fastcall *v45)(_QWORD *, __int64); // [rsp+A8h] [rbp-58h]
  _QWORD v46[2]; // [rsp+B0h] [rbp-50h] BYREF
  void (__fastcall *v47)(_QWORD *, _QWORD *, __int64); // [rsp+C0h] [rbp-40h]
  void (__fastcall *v48)(_QWORD *, __int64); // [rsp+C8h] [rbp-38h]

  v3 = *(void **)(a1 + 24);
  v4 = *(_QWORD *)(a2 + 232);
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v3 <= 9u )
  {
    sub_16E7EE0(a1, "digraph {\n", 0xAu);
    v5 = *(_WORD **)(a1 + 24);
  }
  else
  {
    qmemcpy(v3, "digraph {\n", 10);
    v5 = (_WORD *)(*(_QWORD *)(a1 + 24) + 10LL);
    *(_QWORD *)(a1 + 24) = v5;
  }
  v30 = v4 + 320;
  v33 = *(_QWORD *)(v4 + 328);
  if ( v33 != v4 + 320 )
  {
    do
    {
      v6 = *(_DWORD *)(v33 + 48);
      if ( *(_QWORD *)(a1 + 16) - (_QWORD)v5 <= 1u )
      {
        v7 = sub_16E7EE0(a1, "\t\"", 2u);
      }
      else
      {
        v7 = a1;
        *v5 = 8713;
        *(_QWORD *)(a1 + 24) += 2LL;
      }
      v8 = v33;
      v9 = v34;
      sub_1DD5B60(v34, v33);
      if ( !v35 )
        goto LABEL_60;
      v36(v34, v7);
      v11 = *(__m128i **)(v7 + 24);
      if ( *(_QWORD *)(v7 + 16) - (_QWORD)v11 <= 0xFu )
      {
        v7 = sub_16E7EE0(v7, "\" [ shape=box ]\n", 0x10u);
        v12 = *(_BYTE **)(v7 + 24);
      }
      else
      {
        *v11 = _mm_load_si128((const __m128i *)&xmmword_430B030);
        v12 = (_BYTE *)(*(_QWORD *)(v7 + 24) + 16LL);
        *(_QWORD *)(v7 + 24) = v12;
      }
      if ( *(_QWORD *)(v7 + 16) <= (unsigned __int64)v12 )
      {
        v7 = sub_16E7DE0(v7, 9);
      }
      else
      {
        *(_QWORD *)(v7 + 24) = v12 + 1;
        *v12 = 9;
      }
      v13 = (unsigned int)(2 * v6);
      v14 = sub_16E7A90(v7, *(unsigned int *)(*(_QWORD *)(a2 + 240) + 4 * v13));
      v15 = *(_QWORD *)(v14 + 24);
      v16 = v14;
      if ( (unsigned __int64)(*(_QWORD *)(v14 + 16) - v15) <= 4 )
      {
        v16 = sub_16E7EE0(v14, " -> \"", 5u);
      }
      else
      {
        *(_DWORD *)v15 = 540945696;
        *(_BYTE *)(v15 + 4) = 34;
        *(_QWORD *)(v14 + 24) += 5LL;
      }
      v8 = v33;
      v9 = v37;
      sub_1DD5B60(v37, v33);
      if ( !v38 )
        goto LABEL_60;
      v39(v37, v16);
      v17 = *(_WORD **)(v16 + 24);
      if ( *(_QWORD *)(v16 + 16) - (_QWORD)v17 <= 1u )
      {
        v16 = sub_16E7EE0(v16, "\"\n", 2u);
        v18 = *(_WORD **)(v16 + 24);
        if ( *(_QWORD *)(v16 + 16) - (_QWORD)v18 > 1u )
        {
LABEL_16:
          *v18 = 8713;
          *(_QWORD *)(v16 + 24) += 2LL;
          goto LABEL_17;
        }
      }
      else
      {
        *v17 = 2594;
        v18 = (_WORD *)(*(_QWORD *)(v16 + 24) + 2LL);
        v19 = *(_QWORD *)(v16 + 16);
        *(_QWORD *)(v16 + 24) = v18;
        if ( (unsigned __int64)(v19 - (_QWORD)v18) > 1 )
          goto LABEL_16;
      }
      v16 = sub_16E7EE0(v16, "\t\"", 2u);
LABEL_17:
      v8 = v33;
      v9 = v40;
      sub_1DD5B60(v40, v33);
      if ( !v41 )
        goto LABEL_60;
      v42(v40, v16);
      v20 = *(_QWORD *)(v16 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v16 + 16) - v20) <= 4 )
      {
        v16 = sub_16E7EE0(v16, "\" -> ", 5u);
      }
      else
      {
        *(_DWORD *)v20 = 1043144738;
        *(_BYTE *)(v20 + 4) = 32;
        *(_QWORD *)(v16 + 24) += 5LL;
      }
      v21 = sub_16E7A90(v16, *(unsigned int *)(*(_QWORD *)(a2 + 240) + 4LL * (unsigned int)(v13 + 1)));
      v22 = *(_BYTE **)(v21 + 24);
      if ( (unsigned __int64)v22 >= *(_QWORD *)(v21 + 16) )
      {
        sub_16E7DE0(v21, 10);
      }
      else
      {
        *(_QWORD *)(v21 + 24) = v22 + 1;
        *v22 = 10;
      }
      if ( v41 )
        v41(v40, v40, 3);
      if ( v38 )
        v38(v37, v37, 3);
      if ( v35 )
        v35(v34, v34, 3);
      v23 = *(__int64 **)(v33 + 88);
      v32 = *(__int64 **)(v33 + 96);
      if ( v23 != v32 )
      {
        while ( 1 )
        {
          v28 = *(_WORD **)(a1 + 24);
          if ( *(_QWORD *)(a1 + 16) - (_QWORD)v28 > 1u )
          {
            v24 = a1;
            *v28 = 8713;
            *(_QWORD *)(a1 + 24) += 2LL;
          }
          else
          {
            v24 = sub_16E7EE0(a1, "\t\"", 2u);
          }
          v8 = v33;
          v9 = v43;
          sub_1DD5B60(v43, v33);
          if ( !v44 )
            break;
          v45(v43, v24);
          v25 = *(_QWORD *)(v24 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(v24 + 16) - v25) <= 5 )
          {
            v24 = sub_16E7EE0(v24, "\" -> \"", 6u);
          }
          else
          {
            *(_DWORD *)v25 = 1043144738;
            *(_WORD *)(v25 + 4) = 8736;
            *(_QWORD *)(v24 + 24) += 6LL;
          }
          v8 = *v23;
          v9 = v46;
          sub_1DD5B60(v46, *v23);
          if ( !v47 )
            break;
          v48(v46, v24);
          v26 = *(__m128i **)(v24 + 24);
          if ( *(_QWORD *)(v24 + 16) - (_QWORD)v26 <= 0x15u )
          {
            sub_16E7EE0(v24, "\" [ color=lightgray ]\n", 0x16u);
          }
          else
          {
            si128 = _mm_load_si128((const __m128i *)&xmmword_430B040);
            v26[1].m128i_i32[0] = 544825714;
            v26[1].m128i_i16[2] = 2653;
            *v26 = si128;
            *(_QWORD *)(v24 + 24) += 22LL;
          }
          if ( v47 )
            v47(v46, v46, 3);
          if ( v44 )
            v44(v43, v43, 3);
          if ( v32 == ++v23 )
            goto LABEL_46;
        }
LABEL_60:
        sub_4263D6(v9, v8, v10);
      }
LABEL_46:
      v5 = *(_WORD **)(a1 + 24);
      v33 = *(_QWORD *)(v33 + 8);
    }
    while ( v30 != v33 );
  }
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v5 <= 1u )
  {
    sub_16E7EE0(a1, "}\n", 2u);
  }
  else
  {
    *v5 = 2685;
    *(_QWORD *)(a1 + 24) += 2LL;
  }
  return a1;
}
