// Function: sub_1E41150
// Address: 0x1e41150
//
__int64 __fastcall sub_1E41150(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 result; // rax
  unsigned __int64 v5; // r15
  __int64 *v6; // rbx
  __int64 *v7; // r12
  int v8; // r13d
  _QWORD *v9; // r8
  __int64 v10; // rcx
  __int64 v11; // [rsp+8h] [rbp-78h]
  unsigned int v13; // [rsp+28h] [rbp-58h]
  __int64 v14; // [rsp+30h] [rbp-50h]
  int v16[13]; // [rsp+4Ch] [rbp-34h] BYREF

  result = 8LL * *(unsigned __int8 *)(a2 + 49);
  if ( result )
  {
    v11 = result >> 3;
    v5 = sub_1E0A240(*(_QWORD *)(a1 + 32), result >> 3);
    v6 = *(__int64 **)(a2 + 56);
    if ( v6 == &v6[*(unsigned __int8 *)(a2 + 49)] )
    {
LABEL_12:
      *(_QWORD *)(a2 + 56) = v5;
      *(_BYTE *)(a2 + 49) = v11;
      return a2;
    }
    else
    {
      v7 = &v6[*(unsigned __int8 *)(a2 + 49)];
      v8 = 1;
      while ( 1 )
      {
        while ( 1 )
        {
          v9 = (_QWORD *)*v6;
          v10 = (unsigned int)(v8 - 1);
          if ( (*(_WORD *)(*v6 + 32) & 4) == 0
            && (*(_WORD *)(*v6 + 32) & 0x30) != 0x30
            && (*v9 & 4) == 0
            && (*v9 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            break;
          }
          ++v6;
          *(_QWORD *)(v5 + 8 * v10) = v9;
          ++v8;
          if ( v7 == v6 )
            goto LABEL_12;
        }
        if ( a4 == -1 )
          break;
        v13 = v8 - 1;
        v14 = *v6;
        if ( !(unsigned __int8)sub_1E41020(a1, a3, v16, v10) )
          break;
        ++v6;
        ++v8;
        *(_QWORD *)(v5 + 8LL * v13) = sub_1E0BAE0(
                                        *(_QWORD *)(a1 + 32),
                                        v14,
                                        (unsigned int)(v16[0] * a4),
                                        *(_QWORD *)(v14 + 24));
        if ( v7 == v6 )
          goto LABEL_12;
      }
      *(_QWORD *)(a2 + 56) = 0;
      *(_BYTE *)(a2 + 49) = 0;
      return a2;
    }
  }
  return result;
}
