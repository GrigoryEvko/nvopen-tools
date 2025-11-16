// Function: sub_35EE4F0
// Address: 0x35ee4f0
//
const char *__fastcall sub_35EE4F0(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // rdx
  __int64 v5; // rdx
  const char *result; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  _WORD *v9; // rdx
  __int64 v10; // rdx
  _WORD *v11; // rdx
  unsigned __int8 *v12; // r12
  size_t v13; // rax
  void *v14; // rdi
  size_t v15; // r14
  _WORD *v16; // rdx

  v4 = (unsigned int)a3 >> 28;
  if ( a3 < 0 )
    sub_C64ED0("Bad virtual register encoding", 1u);
  switch ( v4 )
  {
    case 0LL:
      result = sub_35EE460(a3);
      v12 = (unsigned __int8 *)result;
      if ( result )
      {
        v13 = strlen(result);
        v14 = *(void **)(a2 + 32);
        v15 = v13;
        result = (const char *)(*(_QWORD *)(a2 + 24) - (_QWORD)v14);
        if ( v15 > (unsigned __int64)result )
        {
          return (const char *)sub_CB6200(a2, v12, v15);
        }
        else if ( v15 )
        {
          result = (const char *)memcpy(v14, v12, v15);
          *(_QWORD *)(a2 + 32) += v15;
        }
      }
      return result;
    case 1LL:
      v16 = *(_WORD **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v16 <= 1u )
      {
        sub_CB6200(a2, (unsigned __int8 *)"%p", 2u);
      }
      else
      {
        *v16 = 28709;
        *(_QWORD *)(a2 + 32) += 2LL;
      }
      return (const char *)sub_CB59D0(a2, a3 & 0xFFFFFFF);
    case 2LL:
      v8 = *(_QWORD *)(a2 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v8) <= 2 )
      {
        sub_CB6200(a2, "%rs", 3u);
      }
      else
      {
        *(_BYTE *)(v8 + 2) = 115;
        *(_WORD *)v8 = 29221;
        *(_QWORD *)(a2 + 32) += 3LL;
      }
      return (const char *)sub_CB59D0(a2, a3 & 0xFFFFFFF);
    case 3LL:
      v9 = *(_WORD **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v9 <= 1u )
      {
        sub_CB6200(a2, "%r", 2u);
      }
      else
      {
        *v9 = 29221;
        *(_QWORD *)(a2 + 32) += 2LL;
      }
      return (const char *)sub_CB59D0(a2, a3 & 0xFFFFFFF);
    case 4LL:
      v10 = *(_QWORD *)(a2 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v10) <= 2 )
      {
        sub_CB6200(a2, "%rd", 3u);
      }
      else
      {
        *(_BYTE *)(v10 + 2) = 100;
        *(_WORD *)v10 = 29221;
        *(_QWORD *)(a2 + 32) += 3LL;
      }
      return (const char *)sub_CB59D0(a2, a3 & 0xFFFFFFF);
    case 5LL:
      v11 = *(_WORD **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v11 <= 1u )
      {
        sub_CB6200(a2, "%f", 2u);
      }
      else
      {
        *v11 = 26149;
        *(_QWORD *)(a2 + 32) += 2LL;
      }
      return (const char *)sub_CB59D0(a2, a3 & 0xFFFFFFF);
    case 6LL:
      v5 = *(_QWORD *)(a2 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v5) <= 2 )
      {
        sub_CB6200(a2, "%fd", 3u);
      }
      else
      {
        *(_BYTE *)(v5 + 2) = 100;
        *(_WORD *)v5 = 26149;
        *(_QWORD *)(a2 + 32) += 3LL;
      }
      return (const char *)sub_CB59D0(a2, a3 & 0xFFFFFFF);
    case 7LL:
      v7 = *(_QWORD *)(a2 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v7) <= 2 )
      {
        sub_CB6200(a2, "%rq", 3u);
      }
      else
      {
        *(_BYTE *)(v7 + 2) = 113;
        *(_WORD *)v7 = 29221;
        *(_QWORD *)(a2 + 32) += 3LL;
      }
      return (const char *)sub_CB59D0(a2, a3 & 0xFFFFFFF);
  }
}
