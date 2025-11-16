// Function: sub_7EB190
// Address: 0x7eb190
//
void __fastcall sub_7EB190(__int64 a1, __m128i *a2)
{
  __int64 v2; // r12
  int v3; // ecx
  __int64 v4; // rdi
  __int64 v5; // rbx
  __int64 i; // r13
  __int64 v7; // rbx
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 j; // r13
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 k; // rbx
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // r14
  _QWORD *v18; // rax
  __int64 v19; // rax
  _QWORD *v20; // rax
  __int64 v21; // r13
  __int64 v22; // r13
  __int64 v23; // rbx
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __m128i v29; // xmm5
  __int64 v30; // r13
  __int64 v31; // rbx
  __int64 v32; // rsi
  __m128i v33; // [rsp-A8h] [rbp-A8h] BYREF
  __m128i v34; // [rsp-98h] [rbp-98h] BYREF
  __m128i v35; // [rsp-88h] [rbp-88h] BYREF
  __m128i v36; // [rsp-78h] [rbp-78h] BYREF
  __m128i v37; // [rsp-68h] [rbp-68h] BYREF
  __m128i v38; // [rsp-58h] [rbp-58h] BYREF
  __m128i v39; // [rsp-48h] [rbp-48h] BYREF

  if ( (*(_BYTE *)(a1 - 8) & 8) == 0 )
  {
    v2 = a1;
    while ( 2 )
    {
      if ( *(_BYTE *)(v2 + 173) != 12 )
      {
        v3 = *(_DWORD *)(v2 + 64);
        *(_BYTE *)(v2 - 8) |= 8u;
        if ( v3 )
          *(_QWORD *)dword_4F07508 = *(_QWORD *)(v2 + 64);
        if ( (*(_BYTE *)(v2 + 88) & 0x70) == 0x20 )
          *(_BYTE *)(v2 + 88) = *(_BYTE *)(v2 + 88) & 0x8F | 0x30;
        sub_7E2D70(v2);
        v4 = *(_QWORD *)(v2 + 128);
        if ( v4 )
          sub_7EAF80(v4, a2);
        switch ( *(_BYTE *)(v2 + 173) )
        {
          case 1:
          case 3:
          case 8:
          case 0xC:
          case 0xD:
          case 0xE:
          case 0xF:
            return;
          case 2:
            if ( unk_4D047E0 )
            {
              v8 = *(_QWORD *)(v2 + 128);
              if ( (*(_BYTE *)(v8 + 140) & 0xFB) == 8 && (sub_8D4C10(v8, dword_4F077C4 != 2) & 1) != 0 )
                *(_QWORD *)(v2 + 128) = sub_73D4C0(*(const __m128i **)(v2 + 128), dword_4F077C4 == 2);
            }
            return;
          case 4:
            sub_7D8B10((_QWORD *)v2);
            return;
          case 6:
            switch ( *(_BYTE *)(v2 + 176) )
            {
              case 0:
              case 1:
              case 6:
                return;
              case 2:
                v21 = *(_QWORD *)(v2 + 184);
                if ( unk_4D047E0 && *(_BYTE *)(v21 + 173) == 2 )
                  *(_BYTE *)(v2 + 168) |= 8u;
                sub_7EB800(v21);
                if ( (unsigned int)sub_7EBAB0(v21, &v33) )
                  sub_72D580(v33.m128i_i64[0], v2, 1);
                return;
              case 3:
                v30 = *(_QWORD *)(v2 + 184);
                sub_7EB800(v30);
                v31 = sub_8D46C0(*(_QWORD *)(v2 + 128));
                if ( (*(_BYTE *)(v31 + 140) & 0xFB) == 8 && (sub_8D4C10(v31, dword_4F077C4 != 2) & 1) != 0 )
                {
                  v32 = 1;
                  if ( (unsigned int)sub_8D3A70(v31) )
                  {
                    while ( *(_BYTE *)(v31 + 140) == 12 )
                      v31 = *(_QWORD *)(v31 + 160);
                    v32 = ((*(_BYTE *)(v31 + 176) >> 3) ^ 1) & 1;
                  }
                }
                else
                {
                  v32 = 0;
                }
                v33.m128i_i64[0] = sub_7EB890(v30, v32);
                sub_72D580(v33.m128i_i64[0], v2, 1);
                return;
              case 5:
                v22 = *(_QWORD *)(v2 + 128);
                v23 = *(_QWORD *)(v2 + 120);
                v24 = *(_QWORD *)(v2 + 184);
                v33 = _mm_loadu_si128((const __m128i *)v2);
                v34 = _mm_loadu_si128((const __m128i *)(v2 + 16));
                v35 = _mm_loadu_si128((const __m128i *)(v2 + 32));
                v36 = _mm_loadu_si128((const __m128i *)(v2 + 48));
                v37 = _mm_loadu_si128((const __m128i *)(v2 + 64));
                v38 = _mm_loadu_si128((const __m128i *)(v2 + 80));
                v39 = _mm_loadu_si128((const __m128i *)(v2 + 96));
                v25 = sub_7DDA20(v24);
                sub_72D510(v25, v2, 1);
                sub_70FEE0(v2, v22, v26, v27, v28);
                *(__m128i *)v2 = _mm_loadu_si128(&v33);
                *(__m128i *)(v2 + 16) = _mm_loadu_si128(&v34);
                *(__m128i *)(v2 + 32) = _mm_loadu_si128(&v35);
                *(__m128i *)(v2 + 48) = _mm_loadu_si128(&v36);
                *(__m128i *)(v2 + 64) = _mm_loadu_si128(&v37);
                *(__m128i *)(v2 + 80) = _mm_loadu_si128(&v38);
                v29 = _mm_loadu_si128(&v39);
                *(_QWORD *)(v2 + 120) = v23;
                *(__m128i *)(v2 + 96) = v29;
                sub_7607C0(v2, 2);
                return;
              default:
                goto LABEL_41;
            }
          case 7:
            sub_7EAFC0((__m128i *)v2);
            return;
          case 0xA:
            sub_7E31E0(v2);
            sub_802E80(v2, 0, 0, 0);
            if ( (unsigned int)sub_8D2B50(*(_QWORD *)(v2 + 128)) )
              sub_7D8C20((const __m128i *)v2);
            v5 = *(_QWORD *)(v2 + 176);
            for ( i = 0; v5; v5 = *(_QWORD *)(v5 + 120) )
            {
              sub_7EB190(v5);
              if ( (*(_QWORD *)(v5 + 168) & 0xFF0020000000LL) == 0xA0020000000LL )
              {
                v9 = *(_QWORD *)(v5 + 120);
                if ( i )
                  *(_QWORD *)(i + 120) = v9;
                else
                  *(_QWORD *)(v2 + 176) = v9;
              }
              else
              {
                i = v5;
              }
            }
            v7 = *(_QWORD *)(v2 + 128);
            for ( *(_QWORD *)(v2 + 184) = i; *(_BYTE *)(v7 + 140) == 12; v7 = *(_QWORD *)(v7 + 160) )
              ;
            if ( (unsigned int)sub_8D3410(v7) && (unsigned int)sub_7E3130(v7) )
            {
              v10 = *(_QWORD *)(v2 + 176);
              for ( j = 0; v10; v10 = *(_QWORD *)(v10 + 120) )
              {
                if ( *(_BYTE *)(v10 + 173) == 11 )
                  j += *(_QWORD *)(v10 + 184);
                else
                  ++j;
              }
              while ( *(_QWORD *)(v7 + 176) > j )
              {
                v19 = sub_8D4050(v7);
                v20 = sub_7E4750(v19);
                if ( *(_QWORD *)(v2 + 176) )
                  *(_QWORD *)(*(_QWORD *)(v2 + 184) + 120LL) = v20;
                else
                  *(_QWORD *)(v2 + 176) = v20;
                *(_QWORD *)(v2 + 184) = v20;
                ++j;
              }
            }
            else if ( (unsigned __int8)(*(_BYTE *)(v7 + 140) - 9) <= 1u )
            {
              sub_7E3EE0(v7);
              v12 = sub_72FD90(*(_QWORD *)(v7 + 160), 11);
              v13 = *(_QWORD *)(v2 + 176);
              for ( k = v12; v13; k = v15 )
              {
                v15 = sub_72FD90(*(_QWORD *)(k + 112), 11);
                v13 = *(_QWORD *)(v13 + 120);
              }
              if ( k )
              {
                v16 = k;
                v17 = 0;
                do
                {
                  if ( (unsigned int)sub_7E3130(*(_QWORD *)(v16 + 120)) )
                    v17 = v16;
                  v16 = sub_72FD90(*(_QWORD *)(v16 + 112), 11);
                }
                while ( v16 );
                if ( v17 )
                {
                  while ( 1 )
                  {
                    v18 = sub_7E4750(*(_QWORD *)(k + 120));
                    if ( *(_QWORD *)(v2 + 176) )
                      *(_QWORD *)(*(_QWORD *)(v2 + 184) + 120LL) = v18;
                    else
                      *(_QWORD *)(v2 + 176) = v18;
                    *(_QWORD *)(v2 + 184) = v18;
                    if ( k == v17 )
                      break;
                    k = sub_72FD90(*(_QWORD *)(k + 112), 11);
                  }
                }
              }
            }
            return;
          case 0xB:
            v2 = *(_QWORD *)(v2 + 176);
            if ( (*(_BYTE *)(v2 - 8) & 8) != 0 )
              return;
            continue;
          default:
LABEL_41:
            sub_721090();
        }
      }
      break;
    }
  }
}
