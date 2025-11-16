// Function: sub_39E0D20
// Address: 0x39e0d20
//
_BYTE *__fastcall sub_39E0D20(__int64 a1, __int64 a2, _WORD *a3)
{
  _BYTE *result; // rax
  _WORD *v5; // r9
  __int64 v6; // r8
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  char *v10; // r15
  size_t v11; // rax
  size_t v12; // r14
  void *v13; // rdi
  __int64 v14; // rdi
  char *v15; // r15
  __int64 v16; // r14
  char v17; // al
  __int64 v18; // rax
  char v19; // al
  unsigned __int64 v20; // r14
  __int64 v21; // rdi
  __int64 v22; // r15
  char *v23; // rsi
  size_t v24; // rdx
  void *v25; // rdi
  _WORD *v26; // [rsp+0h] [rbp-90h]
  _BYTE *v27; // [rsp+8h] [rbp-88h]
  unsigned __int64 v28; // [rsp+10h] [rbp-80h]
  _WORD *v29; // [rsp+18h] [rbp-78h]
  __int64 v31; // [rsp+28h] [rbp-68h]
  unsigned __int64 v32; // [rsp+28h] [rbp-68h]
  unsigned __int64 v33; // [rsp+30h] [rbp-60h]
  void *v34; // [rsp+40h] [rbp-50h] BYREF
  const char *v35; // [rsp+48h] [rbp-48h]
  char v36; // [rsp+50h] [rbp-40h]

  result = (_BYTE *)(((unsigned __int64)a3 + 3) & 0xFFFFFFFFFFFFFFFCLL);
  v26 = a3;
  v27 = result;
  if ( result )
  {
    v33 = 0;
    while ( 1 )
    {
      v5 = v26;
      v6 = *(_QWORD *)(a1 + 272);
      v7 = v33;
      v8 = v33 + 4;
      v33 += 4LL;
      if ( v33 <= (unsigned __int64)v26 )
        v5 = (_WORD *)v8;
      v9 = *(_QWORD *)(a1 + 280);
      v10 = *(char **)(v9 + 200);
      if ( !v10 )
        goto LABEL_9;
      v28 = v7;
      v29 = v5;
      v31 = *(_QWORD *)(a1 + 272);
      v11 = strlen(*(const char **)(v9 + 200));
      v6 = v31;
      v5 = v29;
      v12 = v11;
      v7 = v28;
      v13 = *(void **)(v31 + 24);
      if ( v11 <= *(_QWORD *)(v31 + 16) - (_QWORD)v13 )
        break;
      sub_16E7EE0(v31, v10, v11);
      v5 = v29;
      v7 = v28;
      v6 = *(_QWORD *)(a1 + 272);
      v32 = (unsigned __int64)v29 - 1;
      if ( v28 < (unsigned __int64)v29 - 1 )
      {
LABEL_10:
        v14 = v6;
        v15 = (char *)(a2 + v7);
        v16 = (__int64)v5 + a2 - 1;
        do
        {
          while ( 1 )
          {
            v17 = *v15;
            v35 = "0x%02x";
            v34 = &unk_49EF3B0;
            v36 = v17;
            v18 = sub_16E8450(v14, (__int64)&v34, (__int64)a3, (__int64)&unk_49EF3B0, v6, (int)v5);
            a3 = *(_WORD **)(v18 + 24);
            if ( *(_QWORD *)(v18 + 16) - (_QWORD)a3 <= 1u )
              break;
            ++v15;
            *a3 = 8236;
            *(_QWORD *)(v18 + 24) += 2LL;
            v14 = *(_QWORD *)(a1 + 272);
            if ( (char *)v16 == v15 )
              goto LABEL_14;
          }
          ++v15;
          sub_16E7EE0(v18, ", ", 2u);
          v14 = *(_QWORD *)(a1 + 272);
        }
        while ( (char *)v16 != v15 );
LABEL_14:
        v6 = v14;
        goto LABEL_15;
      }
LABEL_25:
      v32 = v7;
LABEL_15:
      v35 = "0x%02x";
      v19 = *(_BYTE *)(a2 + v32);
      v34 = &unk_49EF3B0;
      v36 = v19;
      sub_16E8450(v6, (__int64)&v34, (__int64)a3, v7, v6, (int)v5);
      v20 = *(unsigned int *)(a1 + 312);
      if ( *(_DWORD *)(a1 + 312) )
      {
        v22 = *(_QWORD *)(a1 + 272);
        v23 = *(char **)(a1 + 304);
        v24 = *(unsigned int *)(a1 + 312);
        v25 = *(void **)(v22 + 24);
        if ( v20 > *(_QWORD *)(v22 + 16) - (_QWORD)v25 )
        {
          sub_16E7EE0(*(_QWORD *)(a1 + 272), v23, v24);
        }
        else
        {
          memcpy(v25, v23, v24);
          *(_QWORD *)(v22 + 24) += v20;
        }
      }
      *(_DWORD *)(a1 + 312) = 0;
      if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
      {
        result = sub_39E0440(a1);
      }
      else
      {
        v21 = *(_QWORD *)(a1 + 272);
        result = *(_BYTE **)(v21 + 24);
        if ( (unsigned __int64)result >= *(_QWORD *)(v21 + 16) )
        {
          result = (_BYTE *)sub_16E7DE0(v21, 10);
        }
        else
        {
          a3 = result + 1;
          *(_QWORD *)(v21 + 24) = result + 1;
          *result = 10;
        }
      }
      if ( v33 >= (unsigned __int64)v27 )
        return result;
    }
    if ( v11 )
    {
      memcpy(v13, v10, v11);
      v7 = v28;
      v5 = v29;
      *(_QWORD *)(v31 + 24) += v12;
      v6 = *(_QWORD *)(a1 + 272);
    }
LABEL_9:
    v32 = (unsigned __int64)v5 - 1;
    if ( v7 < (unsigned __int64)v5 - 1 )
      goto LABEL_10;
    goto LABEL_25;
  }
  return result;
}
