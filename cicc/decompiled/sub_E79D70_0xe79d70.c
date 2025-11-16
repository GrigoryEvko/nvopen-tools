// Function: sub_E79D70
// Address: 0xe79d70
//
__int64 __fastcall sub_E79D70(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // r15
  _QWORD *i; // r13
  unsigned __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // r14
  _BYTE *v9; // r11
  size_t v10; // r10
  __int64 v11; // rax
  char *v12; // rdi
  size_t v13; // rax
  char *v14; // rax
  __int64 v16; // rax
  _QWORD *v17; // [rsp+8h] [rbp-58h]
  _BYTE *v18; // [rsp+10h] [rbp-50h]
  __int64 v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+18h] [rbp-48h]
  size_t v21; // [rsp+18h] [rbp-48h]
  size_t v22[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = a3;
  v17 = a2;
  if ( a1 != a2 )
  {
    for ( i = a1; v17 != i; i += 12 )
    {
      if ( v3 )
      {
        *(_QWORD *)v3 = *i;
        *(_QWORD *)(v3 + 8) = i[1];
        *(_QWORD *)(v3 + 16) = i[2];
        *(_QWORD *)(v3 + 24) = i[3];
        v5 = i[5] - i[4];
        *(_QWORD *)(v3 + 32) = 0;
        *(_QWORD *)(v3 + 40) = 0;
        *(_QWORD *)(v3 + 48) = 0;
        if ( v5 )
        {
          if ( v5 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_31:
            sub_4261EA(a1, a2, a3);
          a1 = (_QWORD *)v5;
          v6 = sub_22077B0(v5);
        }
        else
        {
          v5 = 0;
          v6 = 0;
        }
        *(_QWORD *)(v3 + 32) = v6;
        *(_QWORD *)(v3 + 40) = v6;
        *(_QWORD *)(v3 + 48) = v6 + v5;
        v7 = i[5];
        if ( v7 != i[4] )
        {
          v8 = i[4];
          while ( 2 )
          {
            if ( !v6 )
              goto LABEL_15;
            *(_QWORD *)v6 = *(_QWORD *)v8;
            *(__m128i *)(v6 + 8) = _mm_loadu_si128((const __m128i *)(v8 + 8));
            *(_QWORD *)(v6 + 24) = *(_QWORD *)(v8 + 24);
            *(_BYTE *)(v6 + 32) = *(_BYTE *)(v8 + 32);
            *(_QWORD *)(v6 + 40) = *(_QWORD *)(v8 + 40);
            a3 = *(_QWORD *)(v8 + 56) - *(_QWORD *)(v8 + 48);
            *(_QWORD *)(v6 + 48) = 0;
            *(_QWORD *)(v6 + 56) = 0;
            *(_QWORD *)(v6 + 64) = 0;
            if ( a3 )
            {
              if ( a3 < 0 )
                goto LABEL_31;
              v19 = a3;
              v11 = sub_22077B0(a3);
              a3 = v19;
              v12 = (char *)v11;
            }
            else
            {
              v12 = 0;
            }
            *(_QWORD *)(v6 + 48) = v12;
            *(_QWORD *)(v6 + 64) = &v12[a3];
            a3 = 0;
            *(_QWORD *)(v6 + 56) = v12;
            a2 = *(_QWORD **)(v8 + 48);
            v13 = *(_QWORD *)(v8 + 56) - (_QWORD)a2;
            if ( v13 )
            {
              v20 = *(_QWORD *)(v8 + 56) - (_QWORD)a2;
              v14 = (char *)memmove(v12, a2, v13);
              a3 = v20;
              v12 = v14;
            }
            *(_QWORD *)(v6 + 56) = &v12[a3];
            a1 = (_QWORD *)(v6 + 88);
            *(_QWORD *)(v6 + 72) = v6 + 88;
            v9 = *(_BYTE **)(v8 + 72);
            v10 = *(_QWORD *)(v8 + 80);
            if ( &v9[v10] && !v9 )
              sub_426248((__int64)"basic_string::_M_construct null not valid");
            v22[0] = *(_QWORD *)(v8 + 80);
            if ( v10 <= 0xF )
            {
              if ( v10 == 1 )
              {
                *(_BYTE *)(v6 + 88) = *v9;
                goto LABEL_14;
              }
              if ( v10 )
              {
LABEL_28:
                a2 = v9;
                memcpy(a1, v9, v10);
                v10 = v22[0];
                a1 = *(_QWORD **)(v6 + 72);
              }
LABEL_14:
              *(_QWORD *)(v6 + 80) = v10;
              *((_BYTE *)a1 + v10) = 0;
LABEL_15:
              v8 += 104;
              v6 += 104;
              if ( v7 == v8 )
                goto LABEL_22;
              continue;
            }
            break;
          }
          v18 = v9;
          v21 = v10;
          v16 = sub_22409D0(v6 + 72, v22, 0);
          v10 = v21;
          v9 = v18;
          *(_QWORD *)(v6 + 72) = v16;
          a1 = (_QWORD *)v16;
          *(_QWORD *)(v6 + 88) = v22[0];
          goto LABEL_28;
        }
LABEL_22:
        *(_QWORD *)(v3 + 40) = v6;
        *(_DWORD *)(v3 + 56) = *((_DWORD *)i + 14);
        *(_DWORD *)(v3 + 60) = *((_DWORD *)i + 15);
        *(_DWORD *)(v3 + 64) = *((_DWORD *)i + 16);
        *(_QWORD *)(v3 + 72) = i[9];
        *(_BYTE *)(v3 + 80) = *((_BYTE *)i + 80);
        *(_BYTE *)(v3 + 81) = *((_BYTE *)i + 81);
        *(_DWORD *)(v3 + 84) = *((_DWORD *)i + 21);
        *(_BYTE *)(v3 + 88) = *((_BYTE *)i + 88);
        *(_BYTE *)(v3 + 89) = *((_BYTE *)i + 89);
      }
      v3 += 96;
    }
  }
  return v3;
}
