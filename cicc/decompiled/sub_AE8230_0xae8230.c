// Function: sub_AE8230
// Address: 0xae8230
//
unsigned __int64 __fastcall sub_AE8230(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int64 result; // rax
  unsigned __int8 v4; // al
  unsigned __int8 *v5; // r13
  unsigned __int8 *v6; // rdx
  unsigned __int8 v7; // al
  unsigned __int8 *v8; // r13
  unsigned __int8 v9; // dl
  __int64 v10; // rbx
  unsigned __int64 i; // r13
  unsigned __int8 v12; // al
  unsigned __int8 *v13; // r13
  unsigned __int8 v14; // dl
  unsigned __int8 **v15; // rbx
  unsigned __int8 **v16; // r13
  __int64 v17; // r14

  while ( 1 )
  {
    result = sub_AE7B90(a1, (__int64)a2);
    if ( !(_BYTE)result )
      break;
    v4 = *(a2 - 16);
    v5 = a2 - 16;
    if ( (v4 & 2) != 0 )
      v6 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
    else
      v6 = &v5[-8 * ((v4 >> 2) & 0xF)];
    sub_AE8080(a1, *((unsigned __int8 **)v6 + 1));
    result = *a2;
    if ( (_BYTE)result == 15 )
    {
      v7 = *(a2 - 16);
      if ( (v7 & 2) != 0 )
        v8 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
      else
        v8 = &v5[-8 * ((v7 >> 2) & 0xF)];
      result = *((_QWORD *)v8 + 3);
      if ( result )
      {
        v9 = *(_BYTE *)(result - 16);
        if ( (v9 & 2) != 0 )
        {
          v10 = *(_QWORD *)(result - 32);
          result = *(unsigned int *)(result - 24);
        }
        else
        {
          v10 = result - 16 - 8LL * ((v9 >> 2) & 0xF);
          result = (*(_WORD *)(result - 16) >> 6) & 0xF;
        }
        for ( i = v10 + 8 * result; i != v10; result = sub_AE8230(a1) )
          v10 += 8;
      }
      return result;
    }
    if ( (_BYTE)result == 14 )
    {
      sub_AE8230(a1);
      v12 = *(a2 - 16);
      if ( (v12 & 2) != 0 )
        v13 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
      else
        v13 = &v5[-8 * ((v12 >> 2) & 0xF)];
      result = *((_QWORD *)v13 + 4);
      if ( result )
      {
        v14 = *(_BYTE *)(result - 16);
        if ( (v14 & 2) != 0 )
        {
          v15 = *(unsigned __int8 ***)(result - 32);
          result = *(unsigned int *)(result - 24);
        }
        else
        {
          v15 = (unsigned __int8 **)(result - 16 - 8LL * ((v14 >> 2) & 0xF));
          result = (*(_WORD *)(result - 16) >> 6) & 0xF;
        }
        v16 = &v15[result];
        if ( v16 != v15 )
        {
          v17 = 0x140000F000LL;
          do
          {
            result = **v15;
            if ( (unsigned __int8)result <= 0x24u && _bittest64(&v17, result) )
            {
              result = sub_AE8230(a1);
            }
            else if ( (_BYTE)result == 18 )
            {
              result = sub_AE8440(a1);
            }
            ++v15;
          }
          while ( v16 != v15 );
        }
      }
      return result;
    }
    if ( (_BYTE)result != 13 )
      return result;
    a2 = (unsigned __int8 *)*((_QWORD *)sub_A17150(a2 - 16) + 3);
  }
  return result;
}
