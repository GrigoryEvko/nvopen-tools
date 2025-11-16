// Function: sub_3568E80
// Address: 0x3568e80
//
__int64 __fastcall sub_3568E80(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // r14
  unsigned __int64 *v6; // r12
  __int64 result; // rax
  unsigned __int64 *v8; // r15
  _QWORD *v9; // r8
  unsigned __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rdx
  _QWORD *v14; // [rsp+0h] [rbp-40h]
  unsigned __int64 v15[7]; // [rsp+8h] [rbp-38h] BYREF

  v15[0] = a2;
  v5 = a1[4];
  sub_3568D60(a3, v15);
  sub_3567E40(a1, v15[0]);
  v6 = *(unsigned __int64 **)(v15[0] + 112);
  result = *(unsigned int *)(v15[0] + 120);
  v8 = &v6[result];
  if ( v6 != v8 )
  {
    v9 = a3 + 1;
    do
    {
      v10 = *v6;
      if ( *v6 != v5 )
      {
        result = a3[2];
        if ( !result )
          goto LABEL_11;
        v11 = (__int64)v9;
        do
        {
          while ( 1 )
          {
            v12 = *(_QWORD *)(result + 16);
            v13 = *(_QWORD *)(result + 24);
            if ( *(_QWORD *)(result + 32) >= v10 )
              break;
            result = *(_QWORD *)(result + 24);
            if ( !v13 )
              goto LABEL_9;
          }
          v11 = result;
          result = *(_QWORD *)(result + 16);
        }
        while ( v12 );
LABEL_9:
        if ( v9 == (_QWORD *)v11 || *(_QWORD *)(v11 + 32) > v10 )
        {
LABEL_11:
          v14 = v9;
          result = sub_3568E80(a1, v10, a3);
          v9 = v14;
        }
      }
      ++v6;
    }
    while ( v8 != v6 );
  }
  return result;
}
