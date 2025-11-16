// Function: sub_A5C960
// Address: 0xa5c960
//
__int64 __fastcall sub_A5C960(__int64 *a1, unsigned int *a2, const void *a3, size_t a4)
{
  __int64 result; // rax
  int *v8; // r12
  __int64 v9; // rdi
  _BYTE *v10; // rax
  __int64 v11; // rsi
  __int64 v12; // r8
  unsigned int v13; // ecx
  __m128i *v14; // rax
  __m128i *v15; // rdi
  __int64 v16; // r10
  __int64 v17; // r9
  __int64 v18; // rdi
  _BYTE *v19; // rax
  __int64 v20; // rax
  _QWORD *v21; // rdi
  unsigned int v22; // [rsp-6Ch] [rbp-6Ch]
  unsigned int v23; // [rsp-68h] [rbp-68h]
  __int64 v24; // [rsp-68h] [rbp-68h]
  __int64 v25; // [rsp-68h] [rbp-68h]
  __int64 v26; // [rsp-68h] [rbp-68h]
  __int64 v27; // [rsp-60h] [rbp-60h]
  __int64 v28[11]; // [rsp-58h] [rbp-58h] BYREF

  result = a2[2];
  if ( (_DWORD)result )
  {
    if ( !*((_DWORD *)a1 + 92) )
    {
      v21 = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 8LL) + 8LL) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 8LL) + 8LL) & 4) != 0 )
        v21 = (_QWORD *)*v21;
      sub_B6F6A0(v21, a1 + 45);
      result = a2[2];
    }
    v8 = *(int **)a2;
    v28[0] = (__int64)off_4979428;
    result = *(_QWORD *)a2 + 16 * result;
    v28[1] = (__int64)(a1 + 5);
    v28[2] = a1[4];
    v28[3] = a1[1];
    v27 = result;
    if ( result != *(_QWORD *)a2 )
    {
      while ( 1 )
      {
        v12 = *a1;
        v13 = *v8;
        v14 = *(__m128i **)(*a1 + 24);
        v15 = *(__m128i **)(*a1 + 32);
        v16 = *a1;
        if ( (char *)v14 - (char *)v15 < a4 )
        {
          v23 = *v8;
          sub_CB6200(*a1, a3, a4);
          v12 = *a1;
          v13 = v23;
          v14 = *(__m128i **)(*a1 + 24);
          v15 = *(__m128i **)(*a1 + 32);
          v16 = *a1;
        }
        else if ( a4 )
        {
          v22 = *v8;
          v24 = *a1;
          memcpy(v15, a3, a4);
          v13 = v22;
          *(_QWORD *)(v24 + 32) += a4;
          v12 = *a1;
          v14 = *(__m128i **)(*a1 + 24);
          v15 = *(__m128i **)(*a1 + 32);
          v16 = *a1;
        }
        v17 = v13;
        if ( v13 < *((_DWORD *)a1 + 92) )
          break;
        if ( (unsigned __int64)((char *)v14 - (char *)v15) <= 0xF )
        {
          v26 = v13;
          v20 = sub_CB6200(v12, "!<unknown kind #", 16);
          v17 = v26;
          v16 = v20;
        }
        else
        {
          *v15 = _mm_load_si128((const __m128i *)&xmmword_3F24B10);
          *(_QWORD *)(v12 + 32) += 16LL;
        }
        v18 = sub_CB59D0(v16, v17);
        v19 = *(_BYTE **)(v18 + 32);
        if ( *(_BYTE **)(v18 + 24) == v19 )
        {
          sub_CB6200(v18, ">", 1);
          goto LABEL_8;
        }
        *v19 = 62;
        ++*(_QWORD *)(v18 + 32);
        v9 = *a1;
        v10 = *(_BYTE **)(*a1 + 32);
        if ( (unsigned __int64)v10 >= *(_QWORD *)(*a1 + 24) )
        {
LABEL_19:
          sub_CB5D20(v9, 32);
          goto LABEL_10;
        }
LABEL_9:
        *(_QWORD *)(v9 + 32) = v10 + 1;
        *v10 = 32;
LABEL_10:
        v11 = *((_QWORD *)v8 + 1);
        v8 += 4;
        result = (__int64)sub_A5C090(*a1, v11, v28);
        if ( (int *)v27 == v8 )
          return result;
      }
      if ( v15 == v14 )
      {
        v25 = v13;
        sub_CB6200(v12, &unk_3F6A4C5, 1);
        v17 = v25;
      }
      else
      {
        v15->m128i_i8[0] = 33;
        ++*(_QWORD *)(v12 + 32);
      }
      sub_A518E0(*(unsigned __int8 **)(a1[45] + 16 * v17), *(_QWORD *)(a1[45] + 16 * v17 + 8), *a1);
LABEL_8:
      v9 = *a1;
      v10 = *(_BYTE **)(*a1 + 32);
      if ( (unsigned __int64)v10 >= *(_QWORD *)(*a1 + 24) )
        goto LABEL_19;
      goto LABEL_9;
    }
  }
  return result;
}
