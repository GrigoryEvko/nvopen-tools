// Function: sub_2C3AB70
// Address: 0x2c3ab70
//
void __fastcall sub_2C3AB70(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 *v4; // rdx
  unsigned __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // r15
  char *v11; // rdi
  __int64 *v12; // rsi
  __int64 v13; // rax
  char *v14; // [rsp+0h] [rbp-70h] BYREF
  int v15; // [rsp+8h] [rbp-68h]
  char v16; // [rsp+10h] [rbp-60h] BYREF

  v1 = sub_2BF3F10(a1);
  v2 = sub_2BF04D0(v1);
  v3 = sub_2BF05A0(v2);
  v8 = *(_QWORD *)(v2 + 120);
  if ( v8 != v3 )
  {
    v9 = v3;
    v10 = 8860176;
    do
    {
      while ( 1 )
      {
        if ( !v8 )
          BUG();
        if ( *(_BYTE *)(v8 - 16) == 36 && (unsigned int)(*(_DWORD *)(*(_QWORD *)(v8 + 128) + 40LL) - 1) <= 1 )
        {
          sub_2C3A0F0(&v14, v8 + 72, (__int64)v4, v5, v6, v7);
          v11 = v14;
          v12 = (__int64 *)&v14[8 * v15];
          v4 = (__int64 *)v14;
          if ( v12 != (__int64 *)v14 )
          {
            do
            {
              v13 = *v4;
              if ( *v4 )
              {
                v5 = *(unsigned __int8 *)(v13 - 32);
                if ( (unsigned __int8)v5 <= 0x17u )
                {
                  if ( _bittest64(&v10, v5) )
                  {
                    v5 = *(unsigned __int8 *)(v13 + 112);
                    switch ( *(_BYTE *)(v13 + 112) )
                    {
                      case 1:
                        *(_BYTE *)(v13 + 116) &= 0xFCu;
                        break;
                      case 2:
                      case 3:
                      case 6:
                        *(_BYTE *)(v13 + 116) &= ~1u;
                        break;
                      case 4:
                        *(_DWORD *)(v13 + 116) = 0;
                        break;
                      case 5:
                        *(_BYTE *)(v13 + 116) &= 0xF9u;
                        break;
                      default:
                        v5 = *(unsigned __int8 *)(v13 - 32);
                        break;
                    }
                  }
                }
              }
              ++v4;
            }
            while ( v12 != v4 );
          }
          if ( v11 != &v16 )
            break;
        }
        v8 = *(_QWORD *)(v8 + 8);
        if ( v8 == v9 )
          return;
      }
      _libc_free((unsigned __int64)v11);
      v8 = *(_QWORD *)(v8 + 8);
    }
    while ( v8 != v9 );
  }
}
