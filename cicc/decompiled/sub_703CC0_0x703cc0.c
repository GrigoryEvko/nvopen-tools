// Function: sub_703CC0
// Address: 0x703cc0
//
const char *__fastcall sub_703CC0(__int64 a1)
{
  __int64 *v1; // r15
  __int64 v2; // r13
  int v3; // eax
  int i; // r14d
  __int64 *j; // rbx
  unsigned __int8 v6; // al
  __int64 v7; // r12
  char v8; // cl
  const char *result; // rax
  char v10; // dl
  _DWORD *v11; // r12
  unsigned __int64 v12; // rbx
  char v13; // dl
  unsigned int v14; // ecx
  char v15; // dl
  const char *v16; // rdx
  char v17; // al
  unsigned __int64 v18; // rcx
  _QWORD *v19; // rsi
  _QWORD *v20; // rax
  int v21; // [rsp+1Ch] [rbp-1514h]
  const char *v22[222]; // [rsp+28h] [rbp-1508h] BYREF
  __int16 v23; // [rsp+718h] [rbp-E18h]
  _BYTE v24[1768]; // [rsp+720h] [rbp-E10h] BYREF
  __int16 v25; // [rsp+E08h] [rbp-728h]
  _BYTE v26[1768]; // [rsp+E10h] [rbp-720h] BYREF
  __int16 v27; // [rsp+14F8h] [rbp-38h]
  _WORD v28[24]; // [rsp+1500h] [rbp-30h] BYREF

  v1 = *(__int64 **)(a1 + 144);
  memset(&v22[1], 0, 0x6E8u);
  v2 = *(_QWORD *)(a1 + 160);
  v23 = 0;
  memset(v24, 0, sizeof(v24));
  v25 = 0;
  memset(v26, 0, sizeof(v26));
  v27 = 0;
  if ( (_DWORD)v2 )
  {
    if ( (int)v2 <= 0 )
      goto LABEL_16;
    v3 = 30;
    if ( (int)v2 <= 30 )
      v3 = v2;
    v21 = v3;
  }
  else
  {
    v21 = 1;
  }
  for ( i = 0; i < v21; ++i )
  {
    for ( j = v1; j; j = (__int64 *)*j )
    {
      v6 = *((_BYTE *)j + 8);
      if ( v6 != 58 )
      {
        v7 = v6;
        v8 = *((_BYTE *)&v28[15 * v6 - 2664] + i);
        if ( v6 && v8 == 1 )
        {
          sub_684B10(0x469u, (_DWORD *)(a1 + 64), (__int64)*(&off_4B6DCE0 + v6));
          v8 = 1;
        }
        *((_BYTE *)&v28[15 * v7 - 2664] + i) = v8 + 1;
      }
    }
  }
LABEL_16:
  result = *(const char **)(a1 + 120);
  if ( result[173] == 2 )
  {
    result = (const char *)*((_QWORD *)result + 23);
    v22[0] = result;
    v10 = *result;
    v11 = (_DWORD *)(a1 + 64);
    if ( *result )
    {
      v12 = 0;
      do
      {
        if ( v10 != 37 )
          goto LABEL_20;
        v13 = result[1];
        if ( v13 == 91 )
          goto LABEL_28;
        if ( !v13 )
          goto LABEL_20;
        if ( result[2] == 91 )
        {
LABEL_28:
          v14 = 0;
          v22[0] = result + 1;
          v15 = result[1];
          if ( v15 != 91 )
          {
            if ( v15 == 108 )
              v14 = (*(_BYTE *)(a1 + 128) & 0x20) != 0;
            v22[0] = result + 2;
          }
          sub_703AC0(v22, *(_QWORD **)(a1 + 136), *(_QWORD **)(a1 + 152), v14, v11);
          result = v22[0];
          goto LABEL_21;
        }
        if ( v13 == 37 )
        {
          result += 2;
          v22[0] = result;
          goto LABEL_21;
        }
        if ( v13 == 108 && (*(_BYTE *)(a1 + 128) & 0x20) != 0 )
        {
          v16 = result + 2;
          v22[0] = result + 2;
          v17 = result[2];
          if ( (unsigned __int8)(v17 - 48) <= 9u )
          {
            v18 = 0;
            do
            {
              v22[0] = ++v16;
              v18 = (char)(v17 - 48) + 10 * v18;
              v17 = *v16;
            }
            while ( (unsigned __int8)(*v16 - 48) <= 9u );
            if ( !v12 )
            {
              v19 = *(_QWORD **)(a1 + 136);
              v20 = *(_QWORD **)(a1 + 152);
              if ( v19 )
              {
                do
                {
                  v19 = (_QWORD *)*v19;
                  ++v12;
                }
                while ( v19 );
                goto LABEL_47;
              }
              if ( !v20 )
              {
LABEL_41:
                sub_6851C0(0x9AEu, v11);
                result = v22[0];
                goto LABEL_21;
              }
              do
              {
                v20 = (_QWORD *)*v20;
                ++v12;
LABEL_47:
                ;
              }
              while ( v20 );
            }
            result = v16;
            if ( v18 >= v12 )
              goto LABEL_41;
          }
          else
          {
            sub_6851C0(0x9ADu, v11);
            result = v22[0];
          }
        }
        else
        {
LABEL_20:
          v22[0] = ++result;
        }
LABEL_21:
        v10 = *result;
      }
      while ( *result );
    }
  }
  return result;
}
