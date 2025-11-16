// Function: sub_1550BA0
// Address: 0x1550ba0
//
void __fastcall sub_1550BA0(__int64 *a1, unsigned int *a2, const char *a3, size_t a4)
{
  __int64 v4; // r13
  int *v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rdi
  _BYTE *v10; // rax
  unsigned __int8 *v11; // rsi
  __int64 v12; // r8
  unsigned int v13; // r13d
  __m128i *v14; // rax
  __m128i *v15; // rdi
  __int64 v16; // r11
  __int64 v17; // r10
  __int64 v18; // rdi
  _BYTE *v19; // rax
  __int64 v20; // rax
  _QWORD *v21; // rdi
  __int64 v22; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+18h] [rbp-38h]

  v4 = a2[2];
  if ( (_DWORD)v4 )
  {
    if ( !*((_DWORD *)a1 + 84) )
    {
      v21 = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 8LL) + 16LL) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 8LL) + 16LL) & 4) != 0 )
        v21 = (_QWORD *)*v21;
      sub_1603160(v21, a1 + 41);
      v4 = a2[2];
    }
    v7 = *(int **)a2;
    v8 = 4 * v4;
    v24 = *(_QWORD *)a2 + v8 * 4;
    if ( &v7[v8] != v7 )
    {
      while ( 1 )
      {
        v12 = *a1;
        v13 = *v7;
        v14 = *(__m128i **)(*a1 + 16);
        v15 = *(__m128i **)(*a1 + 24);
        v16 = *a1;
        if ( (char *)v14 - (char *)v15 < a4 )
        {
          sub_16E7EE0(*a1, a3, a4);
          v12 = *a1;
          v14 = *(__m128i **)(*a1 + 16);
          v15 = *(__m128i **)(*a1 + 24);
          v16 = *a1;
        }
        else if ( a4 )
        {
          v22 = *a1;
          memcpy(v15, a3, a4);
          *(_QWORD *)(v22 + 24) += a4;
          v12 = *a1;
          v14 = *(__m128i **)(*a1 + 16);
          v15 = *(__m128i **)(*a1 + 24);
          v16 = *a1;
        }
        v17 = v13;
        if ( v13 < *((_DWORD *)a1 + 84) )
          break;
        if ( (unsigned __int64)((char *)v14 - (char *)v15) <= 0xF )
        {
          v20 = sub_16E7EE0(v12, "!<unknown kind #", 16);
          v17 = v13;
          v16 = v20;
        }
        else
        {
          *v15 = _mm_load_si128((const __m128i *)&xmmword_3F24B10);
          *(_QWORD *)(v12 + 24) += 16LL;
        }
        v18 = sub_16E7A90(v16, v17);
        v19 = *(_BYTE **)(v18 + 24);
        if ( *(_BYTE **)(v18 + 16) == v19 )
        {
          sub_16E7EE0(v18, ">", 1);
          goto LABEL_8;
        }
        *v19 = 62;
        ++*(_QWORD *)(v18 + 24);
        v9 = *a1;
        v10 = *(_BYTE **)(*a1 + 24);
        if ( (unsigned __int64)v10 >= *(_QWORD *)(*a1 + 16) )
        {
LABEL_19:
          sub_16E7DE0(v9, 32);
          goto LABEL_10;
        }
LABEL_9:
        *(_QWORD *)(v9 + 24) = v10 + 1;
        *v10 = 32;
LABEL_10:
        v11 = (unsigned __int8 *)*((_QWORD *)v7 + 1);
        v7 += 4;
        sub_154F770(*a1, v11, (__int64)(a1 + 5), a1[4], a1[1]);
        if ( (int *)v24 == v7 )
          return;
      }
      if ( v15 == v14 )
      {
        sub_16E7EE0(v12, "!", 1);
        v17 = v13;
      }
      else
      {
        v15->m128i_i8[0] = 33;
        ++*(_QWORD *)(v12 + 24);
      }
      sub_154A520(*(char **)(a1[41] + 16 * v17), *(_QWORD *)(a1[41] + 16 * v17 + 8), *a1);
LABEL_8:
      v9 = *a1;
      v10 = *(_BYTE **)(*a1 + 24);
      if ( (unsigned __int64)v10 >= *(_QWORD *)(*a1 + 16) )
        goto LABEL_19;
      goto LABEL_9;
    }
  }
}
