// Function: sub_8A1930
// Address: 0x8a1930
//
__int64 *__fastcall sub_8A1930(
        __int64 a1,
        __int64 **a2,
        int a3,
        __int64 a4,
        __int64 a5,
        int a6,
        _DWORD *a7,
        __int64 a8,
        __int64 ***a9)
{
  __int64 v12; // r15
  __int64 v13; // rax
  int v14; // r11d
  __int64 v15; // r10
  __int64 v16; // r12
  char v17; // al
  __int64 v18; // rdx
  __int64 v19; // rsi
  char v20; // cl
  __int64 v21; // r15
  __int64 *result; // rax
  __int64 v23; // r10
  int v24; // eax
  _QWORD *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r13
  __int64 **v29; // r12
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 *v33; // r9
  unsigned int v34; // eax
  int v35; // [rsp+8h] [rbp-68h]
  __int64 v36; // [rsp+10h] [rbp-60h]
  __int64 v37; // [rsp+10h] [rbp-60h]
  int v38; // [rsp+18h] [rbp-58h]
  __int64 v39; // [rsp+18h] [rbp-58h]
  __m128i *v42; // [rsp+38h] [rbp-38h] BYREF

  if ( a9 )
    *a9 = 0;
  v12 = *(_QWORD *)(a1 + 88);
  v13 = sub_892920(a1);
  v14 = 0;
  v15 = *(_QWORD *)(v13 + 88);
  v16 = v13;
  if ( (*(_BYTE *)(v13 + 81) & 0x40) != 0 )
  {
    v39 = **(_QWORD **)(v15 + 32);
    v27 = sub_8A4520(*(_QWORD *)(v15 + 104), a3, a4, a5, a6, (_DWORD)a7, a8);
    v14 = v39;
    v16 = *(_QWORD *)v27;
    v15 = *(_QWORD *)(*(_QWORD *)v27 + 88LL);
  }
  v17 = *((_BYTE *)*a2 + 80);
  v18 = (*a2)[11];
  if ( v17 == 3 )
  {
    v19 = **(_QWORD **)(v18 + 168);
  }
  else if ( (unsigned __int8)(v17 - 4) <= 1u )
  {
    v19 = *(_QWORD *)(*(_QWORD *)(v18 + 168) + 168LL);
  }
  else if ( v17 == 7 )
  {
    v19 = **(_QWORD **)(v18 + 216);
  }
  else
  {
    v19 = *(_QWORD *)(v18 + 240);
  }
  v38 = 0;
  v20 = 0;
  if ( (unsigned __int8)(*((_BYTE *)a2 + 140) - 9) <= 2u && *((char *)a2 + 177) < 0 )
  {
    v38 = 1;
    v20 = 1;
    if ( !*(_QWORD *)(v12 + 152) )
    {
      if ( *(_DWORD *)(a8 + 84) )
      {
        v35 = v14;
        v37 = v15;
        v34 = (unsigned int)sub_896D70(v16, a4, 1);
        v15 = v37;
        v14 = v35;
        v20 = 1;
        LODWORD(v19) = v34;
      }
    }
  }
  LODWORD(v21) = 0;
  if ( (*(_BYTE *)(v15 + 160) & 2) == 0 )
    v21 = **(_QWORD **)(v15 + 32);
  if ( HIDWORD(qword_4F077B4)
    && !(_DWORD)qword_4F077B4
    && qword_4F077A8 <= 0x1ADAFu
    && v20
    && (a6 & 0x2020) == 0x2020
    && *(_DWORD *)(a8 + 76) )
  {
    goto LABEL_37;
  }
  v36 = v15;
  v42 = (__m128i *)sub_8A55D0(v16, v19, v21, v14, a3, a4, a5, a6, (__int64)a7, a8);
  result = 0;
  if ( !*a7 )
  {
    v23 = v36;
    if ( *(_BYTE *)(v16 + 80) == 19 && (*(_BYTE *)(*(_QWORD *)(v16 + 88) + 265LL) & 1) != 0 )
    {
      v28 = *(_QWORD *)(*(_QWORD *)(v36 + 176) + 88LL);
      sub_865900(v16);
      v29 = (__int64 **)sub_8A2270(v28, (_DWORD)v42, v21, a5, a6 & 0xFFFFFFAF, (_DWORD)a7, a8);
      sub_864110(v28, (__int64)v42, v30, v31, v32, v33);
      if ( (__int64 **)v28 == v29 )
        v29 = a2;
      result = *v29;
      if ( !*v29 )
        *a9 = v29;
    }
    else
    {
      if ( unk_4F07794 && (v24 = sub_894550(v16, (__int64)v42, 0, 0, 0), v23 = v36, !v24)
        || (v25 = *(_QWORD **)(v23 + 104), (v26 = v25[22]) != 0)
        && (*(_QWORD *)(v26 + 16) || (*(_BYTE *)(*(_QWORD *)(*v25 + 88LL) + 160LL) & 0x20) != 0)
        && !(unsigned int)sub_89A370(v42->m128i_i64)
        && !(unsigned int)sub_8A00C0(v16, v42->m128i_i64, 0) )
      {
LABEL_37:
        *a7 = 1;
        return 0;
      }
      return sub_8A0370(v16, &v42, v38, 0, 0, 0, 1u);
    }
  }
  return result;
}
