// Function: sub_157F670
// Address: 0x157f670
//
unsigned __int64 __fastcall sub_157F670(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  unsigned __int64 v4; // rbx
  unsigned int v5; // r14d
  __int64 v6; // r9
  unsigned __int64 i; // r15
  unsigned int j; // esi
  __int64 v9; // rdx
  __int64 v10; // rcx
  int v11; // [rsp+Ch] [rbp-34h]

  result = sub_157EBA0(a1);
  if ( result )
  {
    v4 = result;
    result = sub_15F4D60(result);
    v11 = result;
    if ( (_DWORD)result )
    {
      v5 = 0;
      do
      {
        result = sub_15F4DF0(v4, v5);
        v6 = *(_QWORD *)(result + 48);
        for ( i = result + 40; i != v6; v6 = *(_QWORD *)(v6 + 8) )
        {
          if ( !v6 )
            BUG();
          if ( *(_BYTE *)(v6 - 8) != 77 )
            break;
          for ( j = *(_DWORD *)(v6 - 4) & 0xFFFFFFF; j; j = *(_DWORD *)(v6 - 4) & 0xFFFFFFF )
          {
            result = 0;
            v9 = 24LL * *(unsigned int *)(v6 + 32) + 8;
            while ( 1 )
            {
              v10 = v6 - 24 - 24LL * j;
              if ( (*(_BYTE *)(v6 - 1) & 0x40) != 0 )
                v10 = *(_QWORD *)(v6 - 32);
              if ( a1 == *(_QWORD *)(v10 + v9) )
                break;
              result = (unsigned int)(result + 1);
              v9 += 8;
              if ( j == (_DWORD)result )
                goto LABEL_15;
            }
            if ( (result & 0x80000000) != 0LL )
              break;
            result = v10 + 8LL * (int)result;
            *(_QWORD *)(result + 24LL * *(unsigned int *)(v6 + 32) + 8) = a2;
          }
LABEL_15:
          ;
        }
        ++v5;
      }
      while ( v11 != v5 );
    }
  }
  return result;
}
