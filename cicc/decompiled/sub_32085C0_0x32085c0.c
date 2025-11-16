// Function: sub_32085C0
// Address: 0x32085c0
//
unsigned __int64 __fastcall sub_32085C0(__int64 a1)
{
  unsigned __int64 result; // rax
  __int64 v2; // r12
  unsigned int v3; // r14d
  __int64 v4; // rax
  unsigned __int8 v5; // dl
  __int64 v6; // rax
  unsigned __int8 v7; // dl
  __int64 v8; // rcx
  unsigned __int8 **v9; // rbx
  unsigned __int8 **v10; // r15
  int v11; // [rsp+4h] [rbp-3Ch]
  __int64 v12; // [rsp+8h] [rbp-38h]

  v12 = sub_BA8DC0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 2488LL), (__int64)"llvm.dbg.cu", 11);
  result = sub_B91A00(v12);
  v11 = result;
  if ( (_DWORD)result )
  {
    v2 = 0x140000F000LL;
    v3 = 0;
    do
    {
      v4 = sub_B91A10(v12, v3);
      v5 = *(_BYTE *)(v4 - 16);
      if ( (v5 & 2) != 0 )
        v6 = *(_QWORD *)(v4 - 32);
      else
        v6 = v4 - 16 - 8LL * ((v5 >> 2) & 0xF);
      result = *(_QWORD *)(v6 + 40);
      if ( result )
      {
        v7 = *(_BYTE *)(result - 16);
        if ( (v7 & 2) != 0 )
        {
          v8 = *(_QWORD *)(result - 32);
          result = *(unsigned int *)(result - 24);
        }
        else
        {
          v8 = result - 16 - 8LL * ((v7 >> 2) & 0xF);
          result = (*(_WORD *)(result - 16) >> 6) & 0xF;
        }
        v9 = (unsigned __int8 **)(v8 + 8 * result);
        if ( v9 != (unsigned __int8 **)v8 )
        {
          v10 = (unsigned __int8 **)v8;
          do
          {
            result = **v10;
            if ( (unsigned __int8)result <= 0x24u )
            {
              if ( _bittest64(&v2, result) )
                result = sub_3206530(a1, *v10, 0);
            }
            ++v10;
          }
          while ( v9 != v10 );
        }
      }
      ++v3;
    }
    while ( v11 != v3 );
  }
  return result;
}
