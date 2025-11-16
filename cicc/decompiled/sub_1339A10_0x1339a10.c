// Function: sub_1339A10
// Address: 0x1339a10
//
__int64 __fastcall sub_1339A10(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, _BOOL8 *a5, __int64 a6, __int64 a7)
{
  __int64 result; // rax
  unsigned __int64 v8; // r14
  char v11; // bl
  _BOOL8 v12; // rax
  _BOOL8 v13; // r8
  _BOOL4 v14; // edi
  unsigned int v15; // eax
  __int64 v16; // rdx
  _BYTE v17[33]; // [rsp+Fh] [rbp-21h]

  result = 1;
  if ( !(a7 | a6) )
  {
    v8 = *(_QWORD *)(a2 + 8);
    result = 14;
    if ( v8 <= 0xFFFFFFFF )
    {
      if ( pthread_mutex_trylock(&stru_4F96C00) )
      {
        sub_130AD90((__int64)&xmmword_4F96BC0);
        byte_4F96C28 = 1;
      }
      ++*((_QWORD *)&xmmword_4F96BF0 + 1);
      if ( a1 != (_QWORD)xmmword_4F96BF0 )
      {
        ++qword_4F96BE8;
        *(_QWORD *)&xmmword_4F96BF0 = a1;
      }
      v11 = *((_BYTE *)sub_1322320(v8) + 4);
      v17[0] = v11;
      byte_4F96C28 = 0;
      pthread_mutex_unlock(&stru_4F96C00);
      if ( a4 && a5 )
      {
        v12 = *a5;
        if ( *a5 )
        {
          *a4 = v11;
          return 0;
        }
        else
        {
          v13 = v12;
          v14 = v12;
          if ( v12 )
          {
            v15 = 0;
            do
            {
              v16 = v15++;
              a4[v16] = v17[v16];
            }
            while ( v15 < v14 );
          }
          *a5 = v13;
          return 22;
        }
      }
      else
      {
        return 0;
      }
    }
  }
  return result;
}
