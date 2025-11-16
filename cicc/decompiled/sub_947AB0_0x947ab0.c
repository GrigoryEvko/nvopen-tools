// Function: sub_947AB0
// Address: 0x947ab0
//
__int64 __fastcall sub_947AB0(_QWORD *a1, __int64 a2, __int64 i, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdi
  unsigned __int8 v9; // al
  __int64 result; // rax
  unsigned __int8 v11; // al
  unsigned __int64 v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // r12
  __m128i *v15; // rax
  __int64 v16; // r15
  __int64 v17; // r14
  __int64 v18; // rcx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v21; // rbx
  _QWORD *v22; // r15
  _QWORD *v23; // r14
  _QWORD *v24; // r12
  __int64 v25; // rsi
  int v26; // eax
  __int64 v27; // [rsp-10h] [rbp-70h]
  __int64 v28; // [rsp+0h] [rbp-60h]
  __int64 v29; // [rsp+8h] [rbp-58h]
  __int64 v30; // [rsp+8h] [rbp-58h]
  __m128i v31[5]; // [rsp+10h] [rbp-50h] BYREF

  while ( 2 )
  {
    if ( !(dword_4D04720 | dword_4D04658) && (*(_WORD *)(a2 + 24) & 0x10FF) != 0x1002 )
    {
      v8 = *a1;
      v31[0].m128i_i64[0] = *(_QWORD *)(a2 + 36);
      if ( v31[0].m128i_i32[0] )
      {
        sub_92FD10(v8, (unsigned int *)v31);
        sub_91CAC0(v31);
      }
    }
    v9 = *(_BYTE *)(a2 + 24);
    if ( v9 == 17 )
    {
      sub_9365F0(v31, *a1, *(unsigned int **)(a2 + 56), 1, a1[2], *((_DWORD *)a1 + 6), *((_BYTE *)a1 + 28));
      return v27;
    }
    else
    {
      if ( v9 <= 0x11u )
      {
        if ( v9 == 1 )
        {
          v11 = *(_BYTE *)(a2 + 56);
          if ( v11 > 0x19u )
          {
            switch ( v11 )
            {
              case 'I':
                return sub_9477E0(a1, a2, i, a4, a5, a6);
              case '[':
                v13 = *(__int64 **)(a2 + 72);
                v14 = v13[2];
                sub_921EA0((__int64)v31, *a1, v13, 0, 0, 0);
                return sub_947E80(*a1, v14, a1[2], *((unsigned int *)a1 + 6), *((unsigned __int8 *)a1 + 28));
              case '\\':
              case '^':
              case '_':
                return sub_947710((__int64)a1, a2);
              case 'g':
                v21 = *(_QWORD *)(*(_QWORD *)(a2 + 72) + 16LL);
                v29 = *(_QWORD *)(a2 + 72);
                v28 = *(_QWORD *)(v21 + 16);
                v22 = (_QWORD *)sub_945CA0(*a1, (__int64)"cond.true", 0, 0);
                v23 = (_QWORD *)sub_945CA0(*a1, (__int64)"cond.false", 0, 0);
                v24 = (_QWORD *)sub_945CA0(*a1, (__int64)"cond.end", 0, 0);
                v25 = v29;
                v30 = *a1;
                v26 = sub_921E00(*a1, v25);
                sub_945D00(v30, v26, (int)v22, (int)v23, 0);
                sub_92FEA0(*a1, v22, 0);
                sub_947AB0(a1, v21);
                sub_92FD90(*a1, (__int64)v24);
                sub_92FEA0(*a1, v23, 0);
                sub_947AB0(a1, v28);
                sub_92FD90(*a1, (__int64)v24);
                return sub_92FEA0(*a1, v24, 0);
              case 'i':
                return sub_926600((__int64)v31);
              case 'p':
                v15 = sub_945C50(*a1, *(_QWORD *)(a2 + 72));
                v16 = *a1;
                v17 = (__int64)v15;
                v19 = sub_91A390(*(_QWORD *)(*a1 + 32LL) + 8LL, *(_QWORD *)a2, 0, v18);
                result = sub_927EF0(v16, v17, v19);
                v20 = a1[2];
                if ( v20 )
                  return sub_9472D0(*a1, result, v20, *((unsigned int *)a1 + 6), *((_BYTE *)a1 + 28));
                return result;
              default:
                goto LABEL_15;
            }
          }
          if ( v11 > 2u )
          {
            switch ( v11 )
            {
              case 3u:
              case 6u:
              case 8u:
                return sub_947710((__int64)a1, a2);
              case 5u:
                if ( a1[2] )
                {
                  v12 = *(_QWORD *)a2;
                  for ( i = *(unsigned __int8 *)(*(_QWORD *)a2 + 140LL); (_BYTE)i == 12; i = *(unsigned __int8 *)(v12 + 140) )
                    v12 = *(_QWORD *)(v12 + 160);
                  if ( (_BYTE)i != 1 )
                    sub_91B8A0("casting aggregate to non-void type is not supported!", (_DWORD *)(a2 + 36), 1);
                }
                goto LABEL_24;
              case 0x19u:
LABEL_24:
                a2 = *(_QWORD *)(a2 + 72);
                continue;
              default:
                break;
            }
          }
        }
        else if ( v9 == 3 )
        {
          return sub_947710((__int64)a1, a2);
        }
LABEL_15:
        sub_91B8A0("unexpected expression with aggregate type!", (_DWORD *)(a2 + 36), 1);
      }
      if ( v9 != 19 )
        goto LABEL_15;
      return sub_927750((__int64)v31, *a1, (__int64 *)a2);
    }
  }
}
