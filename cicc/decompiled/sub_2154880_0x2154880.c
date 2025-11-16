// Function: sub_2154880
// Address: 0x2154880
//
_QWORD *__fastcall sub_2154880(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  __int64 v6; // rbx
  unsigned __int8 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rdx
  _BYTE *v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdx
  _BYTE *v15; // rsi
  _QWORD v17[7]; // [rsp+8h] [rbp-38h] BYREF

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( a3 )
  {
    v4 = *(unsigned int *)(a3 + 8);
    v6 = v4;
    v7 = *(unsigned __int8 **)(a3 - 8 * v4);
    if ( v7 )
    {
      if ( (unsigned int)*v7 - 1 <= 1 )
      {
        v8 = *((_QWORD *)v7 + 17);
        if ( v8 )
        {
          if ( !*(_BYTE *)(v8 + 16) && (unsigned int)v4 > 1 )
          {
            v9 = 1;
            while ( 1 )
            {
              v10 = *(_QWORD *)(a3 + 8 * (v9 - v4));
              v11 = *(unsigned int *)(v10 + 8);
              v17[0] = v10;
              v12 = *(_BYTE **)(v10 - 8 * v11);
              if ( *v12 )
                goto LABEL_9;
              v13 = sub_161E970((__int64)v12);
              if ( v14 == 13 )
              {
                if ( *(_QWORD *)v13 != 0x657079745F636576LL
                  || *(_DWORD *)(v13 + 8) != 1852401759
                  || *(_BYTE *)(v13 + 12) != 116 )
                {
                  goto LABEL_9;
                }
              }
              else if ( v14 != 20
                     || (*(_QWORD *)v13 ^ 0x6F72675F6B726F77LL | *(_QWORD *)(v13 + 8) ^ 0x5F657A69735F7075LL
                      || *(_DWORD *)(v13 + 16) != 1953393000)
                     && (*(_QWORD *)v13 ^ 0x726F775F64716572LL | *(_QWORD *)(v13 + 8) ^ 0x5F70756F72675F6BLL
                      || *(_DWORD *)(v13 + 16) != 1702521203) )
              {
                goto LABEL_9;
              }
              v15 = (_BYTE *)a1[1];
              if ( v15 == (_BYTE *)a1[2] )
                break;
              if ( v15 )
              {
                *(_QWORD *)v15 = v17[0];
                v15 = (_BYTE *)a1[1];
              }
              ++v9;
              a1[1] = v15 + 8;
              if ( v9 == v6 )
                return a1;
LABEL_10:
              v4 = *(unsigned int *)(a3 + 8);
            }
            sub_21546F0((__int64)a1, v15, v17);
LABEL_9:
            if ( ++v9 == v6 )
              return a1;
            goto LABEL_10;
          }
        }
      }
    }
  }
  return a1;
}
