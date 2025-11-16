// Function: sub_8DC200
// Address: 0x8dc200
//
__m128i *__fastcall sub_8DC200(
        __int64 a1,
        unsigned int (__fastcall *a2)(__m128i *, _QWORD, __m128i **),
        unsigned int a3)
{
  const __m128i *v4; // r15
  char v6; // di
  __m128i *v7; // rbx
  unsigned int v9; // eax
  char v10; // al
  __m128i *v11; // r14
  bool v12; // zf
  __int64 v13; // rax
  __int64 v14; // r13
  __m128i *v15; // rax
  __int64 v16; // rdx
  _QWORD *v17; // rcx
  unsigned int v18; // eax
  __m128i *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  _QWORD *v23; // rax
  const __m128i *v24; // rdx
  _QWORD *v25; // rcx
  __m128i *v26; // rax
  __int64 *v27; // r15
  _QWORD *j; // r14
  __m128i *v29; // rdi
  _QWORD *v30; // rax
  _QWORD *v31; // r13
  char v32; // al
  char v33; // al
  __int64 v34; // rdi
  __m128i *v35; // rsi
  __int64 v36; // rsi
  char v37; // al
  __int64 v38; // rdi
  char v39; // al
  __int64 v40; // [rsp+8h] [rbp-68h]
  __m128i *v41; // [rsp+18h] [rbp-58h]
  _QWORD *v42; // [rsp+20h] [rbp-50h]
  __int64 i; // [rsp+28h] [rbp-48h]
  __m128i *v44; // [rsp+30h] [rbp-40h] BYREF
  __int64 v45[7]; // [rsp+38h] [rbp-38h] BYREF

  v4 = (const __m128i *)a1;
  switch ( *(_BYTE *)(a1 + 140) )
  {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 9:
    case 0xA:
    case 0xB:
    case 0xE:
    case 0x13:
    case 0x14:
    case 0x15:
      return (__m128i *)v4;
    case 6:
      if ( a2(*(__m128i **)(a1 + 160), a3, &v44) )
      {
        v10 = *(_BYTE *)(a1 + 168);
        if ( (v10 & 1) != 0 )
        {
          if ( (v10 & 2) != 0 )
            return (__m128i *)sub_72D6A0(v44);
          else
            return (__m128i *)sub_72D600(v44);
        }
        else
        {
          return (__m128i *)sub_72D2E0(v44);
        }
      }
      return (__m128i *)v4;
    case 7:
      v11 = *(__m128i **)(a1 + 160);
      v12 = a2(v11, a3, &v44) == 0;
      v13 = *(_QWORD *)(a1 + 168);
      if ( !v12 )
        v11 = v44;
      v14 = *(_QWORD *)(v13 + 40);
      if ( v14 && a2(*(__m128i **)(v13 + 40), a3, &v44) )
      {
        i = 0;
        v41 = 0;
        v14 = *(_QWORD *)(v44[10].m128i_i64[1] + 40);
      }
      else
      {
        v15 = *(__m128i **)(a1 + 160);
        if ( v15 == v11 || v15 && v11 && dword_4F07588 && (v16 = v11[2].m128i_i64[0], v15[2].m128i_i64[0] == v16) && v16 )
        {
          v17 = **(_QWORD ***)(a1 + 168);
          if ( !v17 )
            return (__m128i *)v4;
          for ( i = 0; ; ++i )
          {
            v42 = v17;
            if ( a2((__m128i *)v17[1], a3, &v44) )
              break;
            v17 = (_QWORD *)*v42;
            if ( !*v42 )
              return (__m128i *)v4;
          }
          v41 = v44;
        }
        else
        {
          i = 0;
          v41 = 0;
        }
      }
      v23 = sub_7259C0(7);
      v23[20] = v11;
      v24 = *(const __m128i **)(a1 + 168);
      v25 = v23;
      v40 = (__int64)v23;
      v26 = (__m128i *)v23[21];
      *v26 = _mm_loadu_si128(v24);
      v26[1] = _mm_loadu_si128(v24 + 1);
      v26[2] = _mm_loadu_si128(v24 + 2);
      v26[3] = _mm_loadu_si128(v24 + 3);
      *(_QWORD *)(v25[21] + 8LL) = 0;
      *(_QWORD *)(v25[21] + 40LL) = v14;
      *(_BYTE *)(v25[21] + 21LL) = (v14 != 0) | *(_BYTE *)(v25[21] + 21LL) & 0xFE;
      v27 = **(__int64 ***)(a1 + 168);
      if ( v27 )
      {
        for ( j = 0; ; j = v31 )
        {
          if ( i )
          {
            v29 = (__m128i *)v27[1];
            --i;
            v44 = v29;
          }
          else if ( v41 )
          {
            v44 = v41;
            v29 = v41;
            v41 = 0;
          }
          else
          {
            v44 = (__m128i *)v27[1];
            a2(v44, a3, &v44);
            v29 = v44;
          }
          v30 = sub_72B0C0((__int64)v29, &dword_4F077C8);
          v31 = v30;
          if ( (v27[4] & 4) != 0 )
          {
            v32 = *((_BYTE *)v30 + 32) | 4;
            *((_BYTE *)v31 + 32) = v32;
            v33 = v27[4] & 8 | v32 & 0xF7;
            *((_BYTE *)v31 + 32) = v33;
            *((_BYTE *)v31 + 32) = v27[4] & 0x10 | v33 & 0xEF;
            v31[6] = v27[6];
            v34 = v27[5];
            if ( v34 )
              v31[5] = sub_73BB50(v34);
          }
          v35 = (__m128i *)v27[1];
          if ( v35 == v44
            || v44 && v35 && dword_4F07588 && (v36 = v35[2].m128i_i64[0], v44[2].m128i_i64[0] == v36) && v36 )
          {
            v39 = v27[4] & 0x40 | v31[4] & 0xBF;
            *((_BYTE *)v31 + 32) = v39;
            *((_BYTE *)v31 + 32) = v27[4] & 0x80 | v39 & 0x7F;
          }
          else
          {
            v37 = sub_8D9650(v31[1]);
            v38 = v31[1];
            *((_BYTE *)v31 + 32) = ((v37 & 1) << 6) | v31[4] & 0xBF;
            *((_BYTE *)v31 + 32) = ((unsigned __int8)sub_8DC1A0(v38) << 7) | v31[4] & 0x7F;
          }
          *((_BYTE *)v31 + 33) = *((_BYTE *)v27 + 33) & 1 | *((_BYTE *)v31 + 33) & 0xFE;
          if ( j )
            *j = v31;
          else
            **(_QWORD **)(v40 + 168) = v31;
          v27 = (__int64 *)*v27;
          if ( !v27 )
            break;
        }
      }
      v4 = (const __m128i *)v40;
      sub_7325D0(v40, &dword_4F077C8);
      return (__m128i *)v4;
    case 8:
      v18 = a2(*(__m128i **)(a1 + 160), a3, &v44);
      v6 = 8;
      if ( v18 )
        goto LABEL_4;
      return (__m128i *)v4;
    case 0xC:
      if ( a2(*(__m128i **)(a1 + 160), a3, &v44) )
        return sub_73CA70(v44, a1);
      return (__m128i *)v4;
    case 0xD:
      a2(*(__m128i **)(a1 + 168), a3, &v44);
      a2(*(__m128i **)(a1 + 160), a3, (__m128i **)v45);
      v19 = *(__m128i **)(a1 + 168);
      if ( v19 != v44 )
      {
        if ( !v19 )
          return (__m128i *)sub_73F0A0(v44, v45[0]);
        if ( !v44 )
          return (__m128i *)sub_73F0A0(v44, v45[0]);
        if ( !dword_4F07588 )
          return (__m128i *)sub_73F0A0(v44, v45[0]);
        v20 = v44[2].m128i_i64[0];
        if ( v19[2].m128i_i64[0] != v20 || !v20 )
          return (__m128i *)sub_73F0A0(v44, v45[0]);
      }
      v21 = *(_QWORD *)(a1 + 160);
      if ( v21 != v45[0] )
      {
        if ( !v21 )
          return (__m128i *)sub_73F0A0(v44, v45[0]);
        if ( !v45[0] )
          return (__m128i *)sub_73F0A0(v44, v45[0]);
        if ( !dword_4F07588 )
          return (__m128i *)sub_73F0A0(v44, v45[0]);
        v22 = *(_QWORD *)(v45[0] + 32);
        if ( *(_QWORD *)(v21 + 32) != v22 || !v22 )
          return (__m128i *)sub_73F0A0(v44, v45[0]);
      }
      return (__m128i *)v4;
    case 0xF:
      if ( !a2(*(__m128i **)(a1 + 160), a3, &v44) )
        return (__m128i *)v4;
      v6 = 15;
LABEL_4:
      v7 = (__m128i *)sub_7259C0(v6);
      sub_73C230(v4, v7);
      v4 = v7;
      v7[10].m128i_i64[0] = (__int64)v44;
      return (__m128i *)v4;
    case 0x10:
      v9 = a2(*(__m128i **)(a1 + 160), a3, &v44);
      v6 = 16;
      if ( v9 )
        goto LABEL_4;
      return (__m128i *)v4;
    default:
      sub_721090();
  }
}
