// Function: sub_E4F660
// Address: 0xe4f660
//
_BYTE *__fastcall sub_E4F660(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  _BYTE *result; // rax
  unsigned __int64 v11; // r12
  __int64 v12; // r8
  unsigned __int64 v13; // r15
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int8 *v16; // rsi
  size_t v17; // rax
  void *v18; // rdi
  unsigned __int64 v19; // r12
  __int64 v20; // rdi
  char v21; // al
  __int64 v22; // rax
  char v23; // al
  unsigned __int64 v24; // r12
  bool v25; // zf
  __int64 v26; // rdi
  __int64 v27; // r8
  unsigned __int8 *v28; // rsi
  void *v29; // rdi
  _BYTE *v31; // [rsp+8h] [rbp-78h]
  __int64 v32; // [rsp+10h] [rbp-70h]
  size_t src; // [rsp+18h] [rbp-68h]
  _QWORD *srca; // [rsp+18h] [rbp-68h]
  unsigned __int64 v35; // [rsp+20h] [rbp-60h]
  void *v36; // [rsp+30h] [rbp-50h] BYREF
  const char *v37; // [rsp+38h] [rbp-48h]
  char v38; // [rsp+40h] [rbp-40h]

  v7 = a3 != 0;
  result = (_BYTE *)(4 * (v7 + ((a3 - v7) >> 2)));
  v31 = result;
  if ( result )
  {
    v35 = 0;
    while ( 1 )
    {
      v11 = a3;
      v12 = *(_QWORD *)(a1 + 304);
      v13 = v35;
      v14 = v35 + 4;
      v35 += 4LL;
      if ( v35 <= a3 )
        v11 = v14;
      v15 = *(_QWORD *)(a1 + 312);
      v16 = *(unsigned __int8 **)(v15 + 224);
      if ( !v16 )
        goto LABEL_9;
      v32 = *(_QWORD *)(a1 + 304);
      v17 = strlen(*(const char **)(v15 + 224));
      v12 = v32;
      v7 = v17;
      v18 = *(void **)(v32 + 32);
      if ( v17 <= *(_QWORD *)(v32 + 24) - (_QWORD)v18 )
        break;
      v19 = v11 - 1;
      sub_CB6200(v32, v16, v17);
      v12 = *(_QWORD *)(a1 + 304);
      if ( v19 > v13 )
      {
LABEL_10:
        v20 = v12;
        do
        {
          while ( 1 )
          {
            v21 = *(_BYTE *)(a2 + v13);
            v37 = "0x%02x";
            v38 = v21;
            v36 = &unk_49DD0D8;
            v22 = sub_CB6620(v20, (__int64)&v36, v7, (__int64)&unk_49DD0D8, v12, a6);
            v7 = *(_QWORD *)(v22 + 32);
            if ( (unsigned __int64)(*(_QWORD *)(v22 + 24) - v7) <= 1 )
              break;
            ++v13;
            *(_WORD *)v7 = 8236;
            *(_QWORD *)(v22 + 32) += 2LL;
            v20 = *(_QWORD *)(a1 + 304);
            if ( v13 >= v19 )
              goto LABEL_14;
          }
          ++v13;
          sub_CB6200(v22, (unsigned __int8 *)", ", 2u);
          v20 = *(_QWORD *)(a1 + 304);
        }
        while ( v13 < v19 );
LABEL_14:
        v12 = v20;
        goto LABEL_15;
      }
LABEL_25:
      v19 = v13;
LABEL_15:
      v23 = *(_BYTE *)(a2 + v19);
      v37 = "0x%02x";
      v38 = v23;
      v36 = &unk_49DD0D8;
      sub_CB6620(v12, (__int64)&v36, v7, a4, v12, a6);
      v24 = *(_QWORD *)(a1 + 344);
      if ( v24 )
      {
        v27 = *(_QWORD *)(a1 + 304);
        v28 = *(unsigned __int8 **)(a1 + 336);
        v29 = *(void **)(v27 + 32);
        if ( v24 > *(_QWORD *)(v27 + 24) - (_QWORD)v29 )
        {
          sub_CB6200(*(_QWORD *)(a1 + 304), v28, *(_QWORD *)(a1 + 344));
        }
        else
        {
          srca = *(_QWORD **)(a1 + 304);
          memcpy(v29, v28, *(_QWORD *)(a1 + 344));
          srca[4] += v24;
        }
      }
      v25 = *(_BYTE *)(a1 + 745) == 0;
      *(_QWORD *)(a1 + 344) = 0;
      if ( v25 )
      {
        v26 = *(_QWORD *)(a1 + 304);
        result = *(_BYTE **)(v26 + 32);
        if ( (unsigned __int64)result >= *(_QWORD *)(v26 + 24) )
        {
          result = (_BYTE *)sub_CB5D20(v26, 10);
        }
        else
        {
          v7 = (__int64)(result + 1);
          *(_QWORD *)(v26 + 32) = result + 1;
          *result = 10;
        }
      }
      else
      {
        result = (_BYTE *)sub_E4D630((__int64 *)a1);
      }
      if ( v35 >= (unsigned __int64)v31 )
        return result;
    }
    if ( v17 )
    {
      src = v17;
      memcpy(v18, v16, v17);
      v7 = src;
      *(_QWORD *)(v32 + 32) += src;
      v12 = *(_QWORD *)(a1 + 304);
    }
LABEL_9:
    v19 = v11 - 1;
    if ( v19 > v13 )
      goto LABEL_10;
    goto LABEL_25;
  }
  return result;
}
