// Function: sub_7DB910
// Address: 0x7db910
//
__int64 __fastcall sub_7DB910(unsigned int a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v3; // rcx
  char v4; // dl
  __int64 v5; // r14
  _QWORD *v6; // rbx
  __int64 *v7; // rbx
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  _QWORD *v12; // rsi
  __int64 v13; // r13
  _QWORD *v14; // rax
  _QWORD *v15; // r12
  _QWORD *v16; // rax
  __int64 v17; // rbx
  const char *v18; // r15
  size_t v19; // rax
  char *v20; // rax
  __int64 v21; // r15
  const char *v22; // r14
  size_t v23; // rax
  char *v24; // rax
  const char *v25; // rsi
  _QWORD *v26; // rsi
  const __m128i *v27; // rax
  __m128i *v28; // rax
  const __m128i *v29; // rax
  __m128i *v30; // rax
  const __m128i *v31; // rax
  __m128i *v32; // rax

  switch ( a1 )
  {
    case 0u:
      ((void (*)(void))sub_7DBE60)();
      goto LABEL_20;
    case 1u:
    case 2u:
    case 3u:
    case 4u:
    case 6u:
    case 7u:
    case 9u:
    case 0xAu:
      if ( a1 != 7 )
        goto LABEL_20;
      if ( !a2 )
      {
        v5 = 1 - ((dword_4D0425C == 0) - 1LL);
        goto LABEL_34;
      }
      v2 = **(__int64 ***)(a2 + 168);
      if ( !v2 )
      {
        v5 = dword_4D0425C != 0;
        goto LABEL_8;
      }
      v3 = 0;
      do
      {
        v4 = *((_BYTE *)v2 + 96);
        v2 = (__int64 *)*v2;
        v3 -= ((v4 & 1) == 0) - 1LL;
      }
      while ( v2 );
      v5 = v3 - ((dword_4D0425C == 0) - 1LL);
      if ( v3 == 1 )
      {
LABEL_34:
        result = qword_4F18998;
        if ( qword_4F18998 )
          return result;
        v17 = sub_7E16B0(10);
        qword_4F18998 = v17;
        v18 = off_4B6D4B8[0];
        v19 = strlen(off_4B6D4B8[0]);
        v20 = (char *)sub_7E1510(v19 + 1);
        *(_QWORD *)(v17 + 8) = v20;
        strcpy(v20, v18);
        *(_BYTE *)(v17 + 88) = *(_BYTE *)(v17 + 88) & 0x8F | 0x20;
        v7 = &qword_4F18998;
LABEL_37:
        sub_7E1CA0(*v7);
        if ( a1 != 8 )
        {
          if ( a1 > 8 )
          {
            if ( a1 - 9 <= 1 )
            {
              v11 = sub_7DB910(8, 0);
              goto LABEL_18;
            }
            goto LABEL_48;
          }
          if ( a1 > 5 )
          {
LABEL_17:
            v11 = sub_7DB910(5, 0);
LABEL_18:
            sub_7E1B70("base");
            switch ( a1 )
            {
              case 1u:
              case 2u:
              case 3u:
              case 4u:
              case 5u:
              case 9u:
                goto LABEL_31;
              case 6u:
                v31 = (const __m128i *)sub_7DB910(5, 0);
                v32 = sub_73C570(v31, 1);
                v11 = sub_72D2E0(v32);
                sub_7E1B70("base_type");
                goto LABEL_31;
              case 7u:
                sub_72BA30(6u);
                sub_7E1B70("flags");
                v12 = sub_72BA30(6u);
                sub_7E1B70("base_count");
                v13 = sub_7DBFA0("base_count", v12);
                v14 = sub_7259C0(8);
                v14[20] = v13;
                v15 = v14;
                v14[22] = v5;
                sub_8D6090(v14);
                v11 = (unsigned __int64)v15;
                sub_7E1B70("base_info");
                goto LABEL_31;
              case 8u:
                v26 = sub_72BA30(6u);
                sub_7E1B70("flags");
                v27 = (const __m128i *)sub_7DBE60("flags", v26);
                v28 = sub_73C570(v27, 1);
                v11 = sub_72D2E0(v28);
                sub_7E1B70("pointee");
                goto LABEL_31;
              case 0xAu:
                v29 = (const __m128i *)sub_7DB910(5, 0);
                v30 = sub_73C570(v29, 1);
                v11 = sub_72D2E0(v30);
                sub_7E1B70("context");
LABEL_31:
                sub_7E1C00(*v7, v11);
                return *v7;
              default:
                goto LABEL_48;
            }
          }
          if ( !a1 )
LABEL_48:
            sub_721090();
        }
        v11 = ((__int64 (*)(void))sub_7DBE60)();
        goto LABEL_18;
      }
LABEL_8:
      v6 = (_QWORD *)qword_4F18950;
      if ( qword_4F18950 )
      {
        while ( sub_8D4490(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v6[1] + 160LL) + 112LL) + 112LL)
                                                 + 112LL)
                                     + 120LL)) != v5 )
        {
          v6 = (_QWORD *)*v6;
          if ( !v6 )
            goto LABEL_32;
        }
        v7 = v6 + 1;
      }
      else
      {
LABEL_32:
        v16 = sub_727DC0();
        *v16 = qword_4F18950;
        qword_4F18950 = (__int64)v16;
        v7 = v16 + 1;
      }
      result = *v7;
      if ( !*v7 )
      {
        *v7 = sub_7E16B0(10);
LABEL_15:
        sub_7DB910(7, 0);
        v9 = qword_4F18998;
        *(_QWORD *)(*v7 + 112) = *(_QWORD *)(qword_4F18998 + 112);
        v10 = *v7;
        *(_QWORD *)(v9 + 112) = *v7;
        if ( !*(_QWORD *)(v10 + 112) )
          *(_QWORD *)(qword_4D03FF0 + 56) = v10;
        goto LABEL_17;
      }
      return result;
    case 5u:
      if ( !qword_4F18990 )
        sub_7DB910(6, 0);
      if ( !qword_4F18998 )
        sub_7DB910(7, 0);
      goto LABEL_20;
    case 8u:
      if ( !qword_4F189A8 )
        sub_7DB910(9, 0);
      if ( !qword_4F189B0 )
        sub_7DB910(10, 0);
LABEL_20:
      v7 = &qword_4F18960[a1];
      result = *v7;
      if ( *v7 )
        return result;
      v21 = sub_7E16B0(10);
      *v7 = v21;
      v22 = (const char *)*(&off_4B6D480 + (int)a1);
      v23 = strlen(v22);
      v24 = (char *)sub_7E1510(v23 + 1);
      v25 = v22;
      v5 = 0;
      *(_QWORD *)(v21 + 8) = v24;
      strcpy(v24, v25);
      *(_BYTE *)(v21 + 88) = *(_BYTE *)(v21 + 88) & 0x8F | 0x20;
      if ( a1 != 7 )
        goto LABEL_37;
      goto LABEL_15;
    default:
      goto LABEL_48;
  }
}
