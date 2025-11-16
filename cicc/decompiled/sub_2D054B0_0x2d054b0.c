// Function: sub_2D054B0
// Address: 0x2d054b0
//
__int64 __fastcall sub_2D054B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  char **v6; // rbx
  __int64 i; // r15
  char *v8; // rdi
  char v9; // al
  __int64 v10; // rdx
  _QWORD *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r9
  _QWORD *v14; // rax
  _QWORD *v15; // r13
  _BYTE *v16; // r12
  unsigned int v18; // eax
  char **v19; // [rsp+8h] [rbp-98h]
  __int64 v20; // [rsp+18h] [rbp-88h]
  _QWORD *v21; // [rsp+20h] [rbp-80h] BYREF
  __int64 v22; // [rsp+28h] [rbp-78h]
  _QWORD v23[14]; // [rsp+30h] [rbp-70h] BYREF

  v5 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v6 = *(char ***)(a2 - 8);
    v19 = &v6[v5];
  }
  else
  {
    v19 = (char **)a2;
    v6 = (char **)(a2 - v5 * 8);
  }
  for ( i = 0; v19 != v6; v6 += 4 )
  {
    v8 = *v6;
    v9 = **v6;
    if ( v9 == 5 )
    {
      v21 = v23;
      ++i;
      LODWORD(v10) = 1;
      v22 = 0x800000001LL;
      v23[0] = v8;
      v11 = v23;
      do
      {
        v12 = (unsigned int)v10;
        v10 = (unsigned int)(v10 - 1);
        v13 = v11[v12 - 1];
        LODWORD(v22) = v10;
        if ( (*(_BYTE *)(v13 + 7) & 0x40) != 0 )
        {
          v14 = *(_QWORD **)(v13 - 8);
          v13 = (__int64)&v14[4 * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF)];
        }
        else
        {
          v14 = (_QWORD *)(v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF));
        }
        if ( v14 != (_QWORD *)v13 )
        {
          v15 = v14;
          do
          {
            v16 = (_BYTE *)*v15;
            if ( *(_BYTE *)*v15 == 5 )
            {
              if ( v10 + 1 > (unsigned __int64)HIDWORD(v22) )
              {
                v20 = v13;
                sub_C8D5F0((__int64)&v21, v23, v10 + 1, 8u, a5, v13);
                v10 = (unsigned int)v22;
                v13 = v20;
              }
              ++i;
              v21[v10] = v16;
              v10 = (unsigned int)(v22 + 1);
              LODWORD(v22) = v22 + 1;
            }
            v15 += 4;
          }
          while ( (_QWORD *)v13 != v15 );
          v11 = v21;
        }
      }
      while ( (_DWORD)v10 );
      if ( v11 != v23 )
        _libc_free((unsigned __int64)v11);
    }
    else if ( v9 == 22 )
    {
      v18 = sub_2D04210((__int64)v8);
      a5 = v18;
      if ( (_BYTE)v18 )
        i += 6;
    }
  }
  return i + (int)sub_CF00B0(a2);
}
