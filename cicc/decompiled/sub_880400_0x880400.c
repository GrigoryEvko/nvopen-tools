// Function: sub_880400
// Address: 0x880400
//
_QWORD *__fastcall sub_880400(__int64 a1)
{
  char v1; // al
  _QWORD *result; // rax
  _QWORD *v3; // r15
  _QWORD *v4; // r14
  _QWORD *v5; // rbx
  __int64 v6; // r13
  unsigned __int8 v7; // al
  __int64 v8; // rdi
  char v9; // dl
  __int64 v10; // rax
  __int64 v11; // r12
  char v12; // dl
  __int64 v13; // rax
  __int64 v14; // r12
  char v15; // dl
  __int64 v16; // rax
  _QWORD *v17; // [rsp+8h] [rbp-38h]

  v1 = *(_BYTE *)(a1 + 80);
  if ( (unsigned __int8)(v1 - 4) <= 1u || v1 == 3 && (unsigned int)sub_8D3A70(*(_QWORD *)(a1 + 88)) )
    v17 = (_QWORD *)(*(_QWORD *)(a1 + 96) + 128LL);
  else
    v17 = *(_QWORD **)(a1 + 96);
  result = v17;
  v3 = (_QWORD *)*v17;
  if ( *v17 )
  {
    while ( 1 )
    {
      v4 = v3;
      v5 = 0;
      do
      {
        v6 = (__int64)v4;
        v4 = (_QWORD *)*v4;
        v7 = *(_BYTE *)(v6 + 16);
        if ( v7 != 2 )
        {
          if ( v7 <= 2u )
          {
            if ( v7 )
            {
              v8 = *(_QWORD *)(v6 + 32);
              v9 = *(_BYTE *)(v8 + 140);
              if ( v9 == 12 )
              {
                v10 = *(_QWORD *)(v6 + 32);
                do
                {
                  v10 = *(_QWORD *)(v10 + 160);
                  v9 = *(_BYTE *)(v10 + 140);
                }
                while ( v9 == 12 );
              }
              if ( v9 )
                sub_7325D0(v8, (_DWORD *)(v6 + 8));
            }
            else
            {
              sub_72AEB0(*(_QWORD *)(v6 + 32), (_DWORD *)(v6 + 8));
            }
            goto LABEL_14;
          }
          if ( v7 != 3 )
            sub_721090();
          v11 = *(_QWORD *)(v6 + 32);
          v12 = *(_BYTE *)(v11 + 140);
          if ( v12 == 12 )
          {
            v13 = *(_QWORD *)(v6 + 32);
            do
            {
              v13 = *(_QWORD *)(v13 + 160);
              v12 = *(_BYTE *)(v13 + 140);
            }
            while ( v12 == 12 );
          }
          if ( v12 && (unsigned int)sub_8D5830(*(_QWORD *)(v11 + 160)) )
          {
            sub_5EB950(8u, 604, *(_QWORD *)(v11 + 160), v6 + 8);
            if ( (_QWORD *)v6 == v3 )
            {
LABEL_28:
              v3 = v4;
              goto LABEL_16;
            }
          }
          else
          {
LABEL_14:
            if ( (_QWORD *)v6 == v3 )
              goto LABEL_28;
          }
          *v5 = v4;
LABEL_16:
          *(_QWORD *)v6 = qword_4F60010;
          qword_4F60010 = v6;
          continue;
        }
        v14 = *(_QWORD *)(v6 + 32);
        v15 = *(_BYTE *)(v14 + 140);
        if ( v15 == 12 )
        {
          v16 = *(_QWORD *)(v6 + 32);
          do
          {
            v16 = *(_QWORD *)(v16 + 160);
            v15 = *(_BYTE *)(v16 + 140);
          }
          while ( v15 == 12 );
        }
        if ( !v15 )
          goto LABEL_14;
        if ( !(unsigned int)sub_8D23B0(*(_QWORD *)(v14 + 160)) )
        {
          *(_QWORD *)dword_4F07508 = *(_QWORD *)(v6 + 8);
          sub_8D6090(v14);
          goto LABEL_14;
        }
        v5 = (_QWORD *)v6;
      }
      while ( v4 );
      if ( !v3 )
      {
        result = v17;
        *v17 = 0;
        return result;
      }
    }
  }
  return result;
}
