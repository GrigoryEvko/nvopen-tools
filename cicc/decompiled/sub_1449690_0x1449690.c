// Function: sub_1449690
// Address: 0x1449690
//
__int64 __fastcall sub_1449690(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 result; // rax
  __int64 v7; // rsi
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // r8
  unsigned __int64 *v11; // r12
  unsigned __int64 *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r15
  __int64 v16; // rcx
  __int64 v17; // r8
  _QWORD *v18; // rax
  _QWORD *v19; // rsi
  int v20; // edx
  int v21; // r10d
  __int64 v22; // [rsp+0h] [rbp-40h]
  _QWORD *v23; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 16);
  result = *(unsigned int *)(v4 + 72);
  if ( (_DWORD)result )
  {
    v7 = *(_QWORD *)(v4 + 56);
    v8 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v7 + 16LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
    {
LABEL_3:
      result = v7 + 16 * result;
      if ( v9 != (__int64 *)result )
      {
        v11 = (unsigned __int64 *)v9[1];
        if ( v11 )
        {
          v22 = a2;
          v23 = 0;
          do
          {
            v12 = (unsigned __int64 *)sub_1443D90(a1, v11, a3);
            v11 = v12;
            if ( !v12 )
              break;
            v15 = *v12;
            if ( !*v12 )
              break;
            if ( (unsigned __int8)sub_1443B70(a1, a2, *v12, v13, v14) )
            {
              v18 = sub_14493C0(a1, a2, v15);
              v19 = v23;
              if ( v23 )
              {
                v23 = v18;
                sub_1448590(v18, v19, 0);
                v22 = v15;
              }
              else
              {
                v22 = v15;
                v23 = v18;
              }
            }
          }
          while ( (unsigned __int8)sub_15CC8F0(*(_QWORD *)(a1 + 8), a2, v15, v16, v17) );
          result = v22;
          if ( v22 != a2 )
            return (__int64)sub_1447CE0(a1, a2, v22, a3);
        }
      }
    }
    else
    {
      v20 = 1;
      while ( v10 != -8 )
      {
        v21 = v20 + 1;
        v8 = (result - 1) & (v20 + v8);
        v9 = (__int64 *)(v7 + 16LL * v8);
        v10 = *v9;
        if ( a2 == *v9 )
          goto LABEL_3;
        v20 = v21;
      }
    }
  }
  return result;
}
