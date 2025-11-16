// Function: sub_3244260
// Address: 0x3244260
//
__int64 __fastcall sub_3244260(
        _BYTE *a1,
        unsigned __int64 **a2,
        __int64 (__fastcall *a3)(__int64, unsigned __int64, unsigned __int64 **),
        __int64 a4)
{
  _BYTE *v4; // r15
  unsigned __int64 *v5; // r14
  _BYTE *v6; // rcx
  unsigned int v8; // eax
  unsigned __int64 *v9; // rdx
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // r8
  int v12; // r14d
  unsigned int v13; // r14d
  __int64 result; // rax
  bool v15; // zf
  unsigned __int64 v16; // r13
  unsigned __int64 v17; // r8
  void (__fastcall *v18)(__int64, _QWORD); // r14
  unsigned int v19; // eax
  unsigned int v20; // eax
  unsigned int v21; // esi
  __int64 v22; // rax
  __int64 v23; // rsi
  unsigned __int64 v24; // rsi
  unsigned __int64 v25; // rsi
  unsigned __int64 *v26; // r14
  unsigned int v27; // eax
  unsigned __int64 *v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rsi
  int v31; // [rsp+0h] [rbp-70h]
  _BYTE *v32; // [rsp+0h] [rbp-70h]
  unsigned __int64 v33; // [rsp+0h] [rbp-70h]
  unsigned __int64 *v34; // [rsp+8h] [rbp-68h]
  char v37; // [rsp+27h] [rbp-49h]
  __int64 v38; // [rsp+28h] [rbp-48h]
  unsigned int v39; // [rsp+28h] [rbp-48h]
  _BYTE *v40; // [rsp+28h] [rbp-48h]
  _BYTE *v41; // [rsp+28h] [rbp-48h]
  __m128i v42; // [rsp+30h] [rbp-40h] BYREF

  v4 = a1;
  v5 = *a2;
  if ( a2[1] != *a2 )
  {
    v34 = 0;
    v6 = a1;
    v37 = 0;
    while ( 1 )
    {
      v38 = (__int64)v6;
      v8 = sub_AF4160(a2);
      v6 = (_BYTE *)v38;
      v9 = &v5[v8];
      *a2 = v9;
      v10 = *v5;
      if ( *v5 - 80 <= 0x1F )
      {
        (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v38 + 8LL))(v38, (unsigned int)v10, 0);
        v6 = (_BYTE *)v38;
      }
      else
      {
        if ( v10 - 112 > 0x1F )
        {
          if ( v10 > 0x9F )
          {
            switch ( v10 )
            {
              case 0x1000uLL:
                v20 = *(unsigned __int16 *)(v38 + 96);
                v21 = *((_DWORD *)v5 + 4) + v5[1] - *(_DWORD *)(v38 + 88);
                if ( (_WORD)v20 && v21 > v20 )
                  v21 = *(unsigned __int16 *)(v38 + 96);
                if ( (*(_BYTE *)(v38 + 100) & 7) == 3 )
                  sub_3243270((_BYTE *)v38);
                sub_32422A0((_QWORD *)v38, v21, *(unsigned __int16 *)(v38 + 98));
                *(_BYTE *)(v38 + 100) &= 0xF8u;
                *(_DWORD *)(v38 + 96) = 0;
                return 1;
              case 0x1001uLL:
                v16 = v5[1];
                v17 = v5[2];
                if ( ((*(_BYTE *)(v38 + 101) >> 1) & 0xFu) > 4
                  && *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v38 + 16) + 208LL) + 3693LL) )
                {
                  v33 = v5[2];
                  (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v38 + 8LL))(v38, 168, 0);
                  v18 = *(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v38 + 40LL);
                  v19 = sub_32441A0(v38, v16, v33);
                  v18(v38, v19);
                  v5 = *a2;
                  v6 = (_BYTE *)v38;
                }
                else if ( v37 && (v24 = v34[1], v24 < (unsigned int)v16) )
                {
                  if ( (_BYTE)v17 == 5 )
                  {
                    sub_3243DF0(v38, v24);
                    v5 = *a2;
                    v6 = (_BYTE *)v38;
                    v37 = 0;
                  }
                  else
                  {
                    v37 = 0;
                    v5 += v8;
                    if ( (_BYTE)v17 == 7 )
                    {
                      sub_3243EC0(v38, v24);
                      v5 = *a2;
                      v6 = (_BYTE *)v38;
                    }
                  }
                }
                else
                {
                  v34 = v5;
                  v5 += v8;
                  v37 = 1;
                }
                goto LABEL_14;
              case 0x1002uLL:
                v15 = *(_BYTE *)(v38 + 103) == 0;
                *(_BYTE *)(v38 + 102) = v5[1];
                if ( !v15 )
                  goto LABEL_20;
                *(_BYTE *)(v38 + 103) = 1;
                v5 = *a2;
                goto LABEL_14;
              case 0x1005uLL:
                result = a3(a4, v5[1], a2);
                v6 = (_BYTE *)v38;
                if ( (_BYTE)result )
                  goto LABEL_20;
                *(_BYTE *)(v38 + 100) &= 0xF8u;
                return result;
              case 0x1006uLL:
              case 0x1007uLL:
                v11 = v5[1];
                v12 = v11 + *((_DWORD *)v5 + 4);
                if ( (*(_BYTE *)(v38 + 100) & 7) == 2 )
                {
                  v31 = v11;
                  (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v38 + 8LL))(v38, 148, 0);
                  (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v38 + 24LL))(
                    v38,
                    ((v12 != 0) + ((v12 - (unsigned int)(v12 != 0)) >> 3)) & 0x1FFFFFFF);
                  LODWORD(v11) = v31;
                  v6 = (_BYTE *)v38;
                }
                v39 = 8 * *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(*((_QWORD *)v6 + 2) + 184LL) + 208LL) + 8LL) - v12;
                v13 = v39 + v11;
                if ( v39 )
                {
                  v32 = v6;
                  (*(void (__fastcall **)(_BYTE *, __int64, _QWORD))(*(_QWORD *)v6 + 8LL))(v6, 16, 0);
                  (*(void (__fastcall **)(_BYTE *, _QWORD))(*(_QWORD *)v32 + 24LL))(v32, v39);
                  (*(void (__fastcall **)(_BYTE *, __int64, _QWORD))(*(_QWORD *)v32 + 8LL))(v32, 36, 0);
                  v6 = v32;
                }
                v40 = v6;
                (*(void (__fastcall **)(_BYTE *, __int64, _QWORD))(*(_QWORD *)v6 + 8LL))(v6, 16, 0);
                (*(void (__fastcall **)(_BYTE *, _QWORD))(*(_QWORD *)v40 + 24LL))(v40, v13);
                (*(void (__fastcall **)(_BYTE *, _QWORD, _QWORD))(*(_QWORD *)v40 + 8LL))(
                  v40,
                  (unsigned int)(v10 == 4102) + 37,
                  0);
                v6 = v40;
LABEL_13:
                v6[100] = v6[100] & 0xF8 | 3;
                v5 = *a2;
                goto LABEL_14;
              default:
                goto LABEL_67;
            }
          }
          if ( v10 <= 5 )
LABEL_67:
            BUG();
          switch ( v10 )
          {
            case 6uLL:
              if ( (*(_BYTE *)(v38 + 100) & 7) == 2 )
                goto LABEL_56;
              v42 = _mm_loadu_si128((const __m128i *)a2);
              if ( v9 == (unsigned __int64 *)v42.m128i_i64[1] )
                goto LABEL_65;
              v26 = &v5[v8];
              break;
            case 0x10uLL:
              sub_3242070(v38, v5[1]);
              v5 = *a2;
              v6 = (_BYTE *)v38;
              goto LABEL_14;
            case 0x11uLL:
              (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v38 + 8LL))(v38, 17, 0);
              v25 = v5[1];
              goto LABEL_45;
            case 0x12uLL:
            case 0x14uLL:
            case 0x1AuLL:
            case 0x1BuLL:
            case 0x1CuLL:
            case 0x1DuLL:
            case 0x1EuLL:
            case 0x20uLL:
            case 0x21uLL:
            case 0x22uLL:
            case 0x24uLL:
            case 0x25uLL:
            case 0x26uLL:
            case 0x27uLL:
            case 0x29uLL:
            case 0x2AuLL:
            case 0x2BuLL:
            case 0x2CuLL:
            case 0x2DuLL:
            case 0x2EuLL:
            case 0x30uLL:
            case 0x97uLL:
              v22 = *(_QWORD *)v38;
              v23 = (unsigned int)v10;
              goto LABEL_37;
            case 0x16uLL:
              v22 = *(_QWORD *)v38;
              v23 = 22;
              goto LABEL_37;
            case 0x18uLL:
              v22 = *(_QWORD *)v38;
              v23 = 24;
              goto LABEL_37;
            case 0x23uLL:
              v29 = *(_QWORD *)v38;
              v30 = 35;
              goto LABEL_60;
            case 0x90uLL:
              v29 = *(_QWORD *)v38;
              v30 = 144;
LABEL_60:
              (*(void (__fastcall **)(__int64, __int64, _QWORD))(v29 + 8))(v38, v30, 0);
              (*(void (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v38 + 24LL))(v38, v5[1]);
              v5 = *a2;
              v6 = (_BYTE *)v38;
              goto LABEL_14;
            case 0x92uLL:
              (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v38 + 8LL))(v38, 146, 0);
              (*(void (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v38 + 24LL))(v38, v5[1]);
              v25 = v5[2];
LABEL_45:
              (*(void (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v38 + 16LL))(v38, v25);
              v5 = *a2;
              v6 = (_BYTE *)v38;
              goto LABEL_14;
            case 0x94uLL:
              (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v38 + 8LL))(v38, 148, 0);
              (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v38 + 32LL))(v38, *((unsigned __int8 *)v5 + 8));
              v5 = *a2;
              v6 = (_BYTE *)v38;
              goto LABEL_14;
            case 0x9FuLL:
              goto LABEL_13;
            default:
              goto LABEL_67;
          }
          do
          {
            v27 = sub_AF4160((unsigned __int64 **)&v42);
            v28 = v26;
            v26 += v27;
            v42.m128i_i64[0] = (__int64)v26;
            if ( *v28 != 6 && *v28 != 4096 )
            {
              v6 = (_BYTE *)v38;
LABEL_56:
              v22 = *(_QWORD *)v6;
              v23 = 6;
LABEL_37:
              v41 = v6;
              (*(void (__fastcall **)(_BYTE *, __int64, _QWORD))(v22 + 8))(v6, v23, 0);
              v5 = *a2;
              v6 = v41;
              goto LABEL_14;
            }
          }
          while ( (unsigned __int64 *)v42.m128i_i64[1] != v26 );
          v6 = (_BYTE *)v38;
LABEL_65:
          v6[100] = v6[100] & 0xF8 | 2;
          v5 = *a2;
          goto LABEL_14;
        }
        sub_32421D0(v38, v10 - 112, v5[1]);
        v6 = (_BYTE *)v38;
      }
LABEL_20:
      v5 = *a2;
LABEL_14:
      if ( v5 == a2[1] )
      {
        v4 = v6;
        break;
      }
    }
  }
  result = 1;
  if ( (v4[100] & 7) == 3 && (v4[101] & 1) == 0 )
  {
    sub_3243270(v4);
    return 1;
  }
  return result;
}
