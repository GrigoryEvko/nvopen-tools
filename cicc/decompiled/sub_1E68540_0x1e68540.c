// Function: sub_1E68540
// Address: 0x1e68540
//
__int64 __fastcall sub_1E68540(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 result; // rax
  __int64 v7; // rsi
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // r8
  unsigned __int64 *v11; // rbx
  unsigned __int64 *v12; // rax
  __int64 v13; // r15
  _QWORD *v14; // rax
  int v15; // edx
  int v16; // r9d
  __int64 v17; // [rsp+8h] [rbp-48h]
  _QWORD *v18; // [rsp+10h] [rbp-40h]
  __int64 v19; // [rsp+18h] [rbp-38h]
  _QWORD *v20; // [rsp+18h] [rbp-38h]

  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 232LL);
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
          v17 = a2;
          v18 = 0;
          do
          {
            v12 = (unsigned __int64 *)sub_1E63260(a1, v11, a3);
            v11 = v12;
            if ( !v12 )
              break;
            v13 = *v12;
            if ( !*v12 )
              break;
            if ( (unsigned __int8)sub_1E62FF0(a1, a2, *v12) )
            {
              v14 = sub_1E68270(a1, a2, v13);
              if ( v18 )
              {
                v20 = v14;
                sub_1E67460(v14, v18, 0);
                v17 = v13;
                v18 = v20;
              }
              else
              {
                v17 = v13;
                v18 = v14;
              }
            }
            v19 = *(_QWORD *)(a1 + 8);
            sub_1E06620(v19);
          }
          while ( sub_1E05550(*(_QWORD *)(v19 + 1312), a2, v13) );
          result = v17;
          if ( v17 != a2 )
            return (__int64)sub_1E66BB0(a1, a2, v17, a3);
        }
      }
    }
    else
    {
      v15 = 1;
      while ( v10 != -8 )
      {
        v16 = v15 + 1;
        v8 = (result - 1) & (v15 + v8);
        v9 = (__int64 *)(v7 + 16LL * v8);
        v10 = *v9;
        if ( a2 == *v9 )
          goto LABEL_3;
        v15 = v16;
      }
    }
  }
  return result;
}
