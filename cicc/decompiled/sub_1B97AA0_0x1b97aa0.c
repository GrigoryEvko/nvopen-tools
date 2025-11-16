// Function: sub_1B97AA0
// Address: 0x1b97aa0
//
__int64 __fastcall sub_1B97AA0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 result; // rax
  _QWORD *v4; // rax
  __int64 v5; // r13
  __int64 v6; // r15
  __int64 v7; // r8
  _QWORD *v8; // rdx
  _QWORD *v9; // rax
  _QWORD *v10; // r14
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  _QWORD *v14; // rdx
  __int64 v15; // [rsp+8h] [rbp-38h]

  v2 = a2;
  result = sub_1B97910(a1, a2);
  if ( !(_BYTE)result )
  {
    while ( 1 )
    {
      v2 = *(_QWORD *)(v2 + 8);
      if ( !v2 )
        return 0;
      v4 = sub_1648700(v2);
      v5 = *(_QWORD *)(a1 + 8);
      v6 = v4[5];
      v7 = (__int64)v4;
      v8 = *(_QWORD **)(v5 + 72);
      v9 = *(_QWORD **)(v5 + 64);
      if ( v8 == v9 )
      {
        v10 = &v9[*(unsigned int *)(v5 + 84)];
        if ( v9 == v10 )
        {
          v14 = *(_QWORD **)(v5 + 64);
        }
        else
        {
          do
          {
            if ( v6 == *v9 )
              break;
            ++v9;
          }
          while ( v10 != v9 );
          v14 = v10;
        }
        goto LABEL_17;
      }
      v15 = v7;
      v10 = &v8[*(unsigned int *)(v5 + 80)];
      v9 = sub_16CC9F0(v5 + 56, v6);
      v7 = v15;
      if ( v6 == *v9 )
        break;
      v11 = *(_QWORD *)(v5 + 72);
      if ( v11 == *(_QWORD *)(v5 + 64) )
      {
        v9 = (_QWORD *)(v11 + 8LL * *(unsigned int *)(v5 + 84));
        v14 = v9;
LABEL_17:
        while ( v14 != v9 && *v9 >= 0xFFFFFFFFFFFFFFFELL )
          ++v9;
        goto LABEL_7;
      }
      v9 = (_QWORD *)(v11 + 8LL * *(unsigned int *)(v5 + 80));
LABEL_7:
      if ( v9 != v10 )
      {
        result = sub_1B97910(a1, v7);
        if ( (_BYTE)result )
          return result;
      }
    }
    v12 = *(_QWORD *)(v5 + 72);
    if ( v12 == *(_QWORD *)(v5 + 64) )
      v13 = *(unsigned int *)(v5 + 84);
    else
      v13 = *(unsigned int *)(v5 + 80);
    v14 = (_QWORD *)(v12 + 8 * v13);
    goto LABEL_17;
  }
  return result;
}
