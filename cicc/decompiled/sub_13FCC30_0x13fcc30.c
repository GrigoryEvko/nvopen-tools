// Function: sub_13FCC30
// Address: 0x13fcc30
//
__int64 __fastcall sub_13FCC30(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 *v3; // r14
  __int64 result; // rax
  __int64 v5; // r13
  __int64 *v6; // r14
  __int64 i; // rdi
  unsigned int v8; // r15d
  __int64 v9; // rbx
  int v10; // r12d
  __int64 v11; // rax
  __int64 *v12; // [rsp+0h] [rbp-40h]

  v2 = sub_13FCB50(a1);
  if ( v2 )
  {
    v11 = sub_157EBA0(v2);
    return sub_1625C10(v11, 18, a2);
  }
  else
  {
    v3 = *(__int64 **)(a1 + 32);
    result = *(_QWORD *)(a1 + 40);
    v5 = *v3;
    v12 = (__int64 *)result;
    if ( v3 != (__int64 *)result )
    {
      v6 = v3 + 1;
      for ( i = v5; ; i = *v6++ )
      {
        v8 = 0;
        v9 = sub_157EBA0(i);
        result = sub_15F4D60(v9);
        v10 = result;
        if ( (_DWORD)result )
        {
          do
          {
            while ( 1 )
            {
              result = sub_15F4DF0(v9, v8);
              if ( v5 == result )
                break;
              if ( v10 == ++v8 )
                goto LABEL_9;
            }
            ++v8;
            result = sub_1625C10(v9, 18, a2);
          }
          while ( v10 != v8 );
        }
LABEL_9:
        if ( v12 == v6 )
          break;
      }
    }
  }
  return result;
}
