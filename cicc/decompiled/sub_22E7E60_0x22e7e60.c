// Function: sub_22E7E60
// Address: 0x22e7e60
//
void __fastcall sub_22E7E60(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        unsigned int a9)
{
  unsigned __int64 v10; // rax
  _QWORD *v11; // rax
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rdi
  _QWORD *v18; // rdx
  __int64 v19; // rdi
  _WORD *v20; // rdx
  unsigned __int64 v21; // r14
  _QWORD *i; // rdi
  _QWORD *v23; // rax
  __int64 v24; // rdi
  _BYTE *v25; // rax
  __int64 v26; // rdi
  _BYTE *v27; // rax
  __int64 v28; // [rsp+8h] [rbp-78h]
  __int64 v29; // [rsp+10h] [rbp-70h]
  size_t v30; // [rsp+28h] [rbp-58h] BYREF
  unsigned __int8 *v31; // [rsp+30h] [rbp-50h] BYREF
  size_t v32; // [rsp+38h] [rbp-48h]
  _QWORD v33[8]; // [rsp+40h] [rbp-40h] BYREF

  v10 = sub_B46EC0(a8, a9);
  v11 = sub_22DDF00(*(_QWORD **)(a7 + 8), v10);
  if ( v11 )
  {
    v12 = (unsigned __int64)v11;
    v29 = *(_QWORD *)a1[1];
    v13 = sub_B46EC0(a8, a9);
    v14 = sub_22DDF00(*(_QWORD **)(a7 + 8), v13);
    if ( (*a2 & 4) == 0 && (*v14 & 4) == 0 )
    {
      v21 = *v14 & 0xFFFFFFFFFFFFFFF8LL;
      v28 = *a2;
      for ( i = (_QWORD *)sub_22DBE80(v29, v21); i; i = (_QWORD *)i[1] )
      {
        v23 = (_QWORD *)i[1];
        if ( !v23 || v21 != (*v23 & 0xFFFFFFFFFFFFFFF8LL) )
        {
          if ( v21 == (*i & 0xFFFFFFFFFFFFFFF8LL) && (unsigned __int8)sub_22DB400(i, v28 & 0xFFFFFFFFFFFFFFF8LL) )
          {
            v30 = 16;
            v31 = (unsigned __int8 *)v33;
            v31 = (unsigned __int8 *)sub_22409D0((__int64)&v31, &v30, 0);
            v33[0] = v30;
            *(__m128i *)v31 = _mm_load_si128((const __m128i *)&xmmword_428CE10);
            v32 = v30;
            v31[v30] = 0;
            goto LABEL_5;
          }
          break;
        }
      }
    }
    v32 = 0;
    v31 = (unsigned __int8 *)v33;
    LOBYTE(v33[0]) = 0;
LABEL_5:
    v15 = *a1;
    v16 = *(_QWORD *)(*a1 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(*a1 + 24) - v16) <= 4 )
    {
      v15 = sub_CB6200(v15, "\tNode", 5u);
    }
    else
    {
      *(_DWORD *)v16 = 1685016073;
      *(_BYTE *)(v16 + 4) = 101;
      *(_QWORD *)(v15 + 32) += 5LL;
    }
    sub_CB5A80(v15, (unsigned __int64)a2);
    v17 = *a1;
    v18 = *(_QWORD **)(*a1 + 32);
    if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v18 <= 7u )
    {
      v17 = sub_CB6200(v17, " -> Node", 8u);
    }
    else
    {
      *v18 = 0x65646F4E203E2D20LL;
      *(_QWORD *)(v17 + 32) += 8LL;
    }
    sub_CB5A80(v17, v12);
    if ( v32 )
    {
      v24 = *a1;
      v25 = *(_BYTE **)(*a1 + 32);
      if ( *(_BYTE **)(*a1 + 24) == v25 )
      {
        v24 = sub_CB6200(v24, (unsigned __int8 *)"[", 1u);
      }
      else
      {
        *v25 = 91;
        ++*(_QWORD *)(v24 + 32);
      }
      v26 = sub_CB6200(v24, v31, v32);
      v27 = *(_BYTE **)(v26 + 32);
      if ( *(_BYTE **)(v26 + 24) == v27 )
      {
        sub_CB6200(v26, (unsigned __int8 *)"]", 1u);
      }
      else
      {
        *v27 = 93;
        ++*(_QWORD *)(v26 + 32);
      }
    }
    v19 = *a1;
    v20 = *(_WORD **)(*a1 + 32);
    if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v20 <= 1u )
    {
      sub_CB6200(v19, (unsigned __int8 *)";\n", 2u);
    }
    else
    {
      *v20 = 2619;
      *(_QWORD *)(v19 + 32) += 2LL;
    }
    if ( v31 != (unsigned __int8 *)v33 )
      j_j___libc_free_0((unsigned __int64)v31);
  }
}
