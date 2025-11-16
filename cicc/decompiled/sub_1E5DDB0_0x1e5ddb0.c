// Function: sub_1E5DDB0
// Address: 0x1e5ddb0
//
__int64 __fastcall sub_1E5DDB0(__int64 a1, _QWORD *a2)
{
  __int64 *v2; // r12
  __int64 *i; // rbx
  __int64 v4; // rsi
  __int64 result; // rax
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // r13
  __int64 v13; // rdx
  unsigned int v14; // r12d
  __int64 v15; // r13
  __int64 *v16; // rbx
  __int64 v17; // r14
  unsigned __int64 v18; // r15
  __int64 v19; // rdx
  __int64 v20; // rax
  unsigned int v21; // eax
  __int16 v22; // cx
  unsigned int v23; // ebx
  bool v24; // dl
  char v25; // al
  int v26; // esi
  unsigned int v27; // edx
  unsigned int v28; // edx
  char v29; // al
  __int64 v30; // [rsp+0h] [rbp-D0h]
  __int64 v32; // [rsp+20h] [rbp-B0h]
  unsigned __int64 v33; // [rsp+28h] [rbp-A8h]
  __int64 v34; // [rsp+30h] [rbp-A0h]
  __int64 v36; // [rsp+48h] [rbp-88h]
  __int16 v37; // [rsp+48h] [rbp-88h]
  int v38; // [rsp+50h] [rbp-80h]
  int v39; // [rsp+54h] [rbp-7Ch]
  __int64 v40; // [rsp+58h] [rbp-78h]
  __int64 v41; // [rsp+68h] [rbp-68h] BYREF
  __m128i v42; // [rsp+70h] [rbp-60h] BYREF
  __int64 v43; // [rsp+80h] [rbp-50h]
  __int64 v44; // [rsp+88h] [rbp-48h]
  __int64 v45; // [rsp+90h] [rbp-40h]

  v2 = (__int64 *)a2[2];
  for ( i = (__int64 *)a2[1]; v2 != i; ++i )
  {
    v4 = *i;
    sub_1E5DDB0(a1, v4);
  }
  result = (__int64)(a2[5] - a2[4]) >> 3;
  if ( (_DWORD)result == 1 )
  {
    v6 = *(_QWORD *)(a1 + 264);
    *(_QWORD *)(a1 + 368) = 0;
    *(_QWORD *)(a1 + 376) = 0;
    *(_DWORD *)(a1 + 392) = 0;
    result = *(_QWORD *)(*(_QWORD *)v6 + 264LL);
    if ( (__int64 (*)())result != sub_1D820E0 )
    {
      result = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, __int64, __int64, _QWORD))result)(
                 v6,
                 *(_QWORD *)a2[4],
                 a1 + 368,
                 a1 + 376,
                 a1 + 384,
                 0);
      if ( !(_BYTE)result )
      {
        v7 = *(_QWORD *)(a1 + 264);
        *(_QWORD *)(a1 + 560) = 0;
        *(_QWORD *)(a1 + 568) = 0;
        result = *(_QWORD *)(*(_QWORD *)v7 + 296LL);
        if ( (__int64 (*)())result != sub_1E40440 )
        {
          result = ((__int64 (__fastcall *)(__int64, _QWORD *, __int64, __int64))result)(v7, a2, a1 + 560, a1 + 568);
          if ( !(_BYTE)result )
          {
            result = sub_1E29990((__int64)a2);
            if ( result )
            {
              v8 = *(__int64 **)(a1 + 8);
              v9 = *v8;
              v10 = v8[1];
              if ( v9 == v10 )
LABEL_44:
                BUG();
              while ( *(_UNKNOWN **)v9 != &unk_4FC450C )
              {
                v9 += 16;
                if ( v10 == v9 )
                  goto LABEL_44;
              }
              v11 = *(_QWORD *)a2[4];
              v34 = *(_QWORD *)(*(_QWORD *)(a1 + 232) + 40LL);
              v32 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(
                                  *(_QWORD *)(v9 + 8),
                                  &unk_4FC450C)
                              + 272);
              v30 = sub_1DD5D10(v11);
              v12 = *(_QWORD *)(v11 + 32);
              if ( v12 == v30 )
                return sub_1E5DB00((_QWORD *)a1, (__int64)a2);
LABEL_15:
              v13 = *(_QWORD *)(v12 + 32);
              v33 = *(_QWORD *)(*(_QWORD *)(v34 + 24) + 16LL * (*(_DWORD *)(v13 + 8) & 0x7FFFFFFF))
                  & 0xFFFFFFFFFFFFFFF8LL;
              v38 = *(_DWORD *)(v12 + 40);
              if ( v38 == 1 )
                goto LABEL_37;
              v40 = v12;
              v14 = 1;
              while ( 1 )
              {
                v15 = v13 + 40LL * v14;
                if ( (*(_DWORD *)v15 & 0xFFF00) == 0 )
                  goto LABEL_19;
                v39 = sub_1E6B9A0(v34, v33, byte_3F871B3, 0);
                v36 = *(_QWORD *)(*(_QWORD *)(v40 + 32) + 40LL * (v14 + 1) + 24);
                v16 = (__int64 *)sub_1DD5EE0(v36);
                sub_1DD6ED0(&v41, v36, (__int64)v16);
                v17 = *(_QWORD *)(v36 + 56);
                v18 = (unsigned __int64)sub_1E0B640(v17, *(_QWORD *)(*(_QWORD *)(a1 + 264) + 8LL) + 960LL, &v41, 0);
                sub_1DD5BA0((__int64 *)(v36 + 16), v18);
                v19 = *v16;
                v20 = *(_QWORD *)v18;
                *(_QWORD *)(v18 + 8) = v16;
                v19 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)v18 = v19 | v20 & 7;
                *(_QWORD *)(v19 + 8) = v18;
                *v16 = v18 | *v16 & 7;
                v42.m128i_i64[0] = 0x10000000;
                v42.m128i_i32[2] = v39;
                v43 = 0;
                v44 = 0;
                v45 = 0;
                sub_1E1A9C0(v18, v17, &v42);
                v21 = *(unsigned __int8 *)(v15 + 3);
                v22 = (*(_DWORD *)v15 >> 8) & 0xFFF;
                if ( (v21 & 0x10) != 0 )
                  break;
                v23 = (v21 >> 3) & 4;
                v24 = (v21 & 0x40) != 0;
                if ( (v21 & 0x40) != 0 )
                {
                  v23 |= 8u;
                  goto LABEL_25;
                }
LABEL_27:
                v25 = *(_BYTE *)(v15 + 4);
                v26 = *(_DWORD *)(v15 + 8);
                if ( (v25 & 1) != 0 )
                  v23 |= 0x20u;
                v27 = v23;
                if ( (v25 & 2) != 0 )
                {
                  BYTE1(v27) = BYTE1(v23) | 1;
                  v23 = v27;
                }
                v28 = v23;
                if ( (v25 & 8) != 0 )
                {
                  LOBYTE(v28) = v23 | 0x80;
                  v23 = v28;
                }
                if ( v26 > 0 )
                {
                  v37 = (*(_DWORD *)v15 >> 8) & 0xFFF;
                  v29 = sub_1E31310(v15);
                  v26 = *(_DWORD *)(v15 + 8);
                  v22 = v37;
                  if ( v29 )
                    BYTE1(v23) |= 2u;
                }
                v42.m128i_i8[0] = 0;
                v42.m128i_i32[2] = v26;
                v43 = 0;
                v44 = 0;
                v45 = 0;
                v42.m128i_i8[3] = (((v23 & 0x18) != 0) << 6)
                                | (32 * ((v23 & 4) != 0)) & 0x3F
                                | v42.m128i_i8[3] & 0xF
                                | (16 * ((v23 & 2) != 0)) & 0x3F
                                | ((unsigned __int8)(v23 >> 9) << 7);
                v42.m128i_i16[1] &= 0xF00Fu;
                v42.m128i_i8[4] = (8 * ((unsigned __int8)v23 >> 7))
                                | ((v23 & 0x20) != 0)
                                | v42.m128i_i8[4] & 0xF0
                                | (2 * (BYTE1(v23) & 1)) & 0xF3;
                v42.m128i_i32[0] = ((v22 & 0xFFF) << 8) | v42.m128i_i32[0] & 0xFFF000FF;
                sub_1E1A9C0(v18, v17, &v42);
                sub_1DC1550(v32, v18, 0);
                sub_1E310D0(v15, v39);
                *(_DWORD *)v15 &= 0xFFF000FF;
                if ( v41 )
                  sub_161E7C0((__int64)&v41, v41);
LABEL_19:
                v14 += 2;
                if ( v38 == v14 )
                {
                  v12 = v40;
LABEL_37:
                  if ( (*(_BYTE *)v12 & 4) == 0 )
                  {
                    while ( (*(_BYTE *)(v12 + 46) & 8) != 0 )
                      v12 = *(_QWORD *)(v12 + 8);
                  }
                  v12 = *(_QWORD *)(v12 + 8);
                  if ( v12 == v30 )
                    return sub_1E5DB00((_QWORD *)a1, (__int64)a2);
                  goto LABEL_15;
                }
                v13 = *(_QWORD *)(v40 + 32);
              }
              v24 = (v21 & 0x40) != 0;
              v23 = (v21 & 0x20) == 0 ? 2 : 6;
LABEL_25:
              if ( v24 && (v21 & 0x10) != 0 )
                v23 |= 0x10u;
              goto LABEL_27;
            }
          }
        }
      }
    }
  }
  return result;
}
