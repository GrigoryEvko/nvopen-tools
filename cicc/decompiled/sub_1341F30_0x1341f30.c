// Function: sub_1341F30
// Address: 0x1341f30
//
unsigned __int64 __fastcall sub_1341F30(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r14
  unsigned __int64 result; // rax
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // r15
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // rdi
  unsigned __int64 *v11; // rcx
  unsigned __int64 v12; // r8
  unsigned __int64 v13; // rdx
  _QWORD *v14; // rax
  unsigned int i; // edx
  __int64 v16; // r9
  _QWORD *v17; // rdx
  _QWORD *v18; // r9
  unsigned __int64 v19; // rax
  __int64 v20; // [rsp+0h] [rbp-1C0h]
  _QWORD v21[54]; // [rsp+10h] [rbp-1B0h] BYREF

  v3 = (_QWORD *)(a1 + 432);
  if ( !a1 )
  {
    v3 = v21;
    v20 = a3;
    sub_130D500(v21);
    a3 = v20;
  }
  result = *(_QWORD *)(a3 + 16) & 0xFFFFFFFFFFFFF000LL;
  if ( result > 0x2000 )
  {
    v6 = *(_QWORD *)(a3 + 8) & 0xFFFFFFFFFFFFF000LL;
    v7 = result + v6 - 0x2000;
    v8 = v6 + 4096;
    if ( v7 >= v8 )
    {
      v9 = v8;
      result = 0;
      do
      {
        if ( v8 == v9 || (v9 & 0x3FFFFFFF) == 0 )
        {
          v10 = v9 & 0xFFFFFFFFC0000000LL;
          v11 = (_QWORD *)((char *)v3 + ((v9 >> 26) & 0xF0));
          v12 = *v11;
          if ( (v9 & 0xFFFFFFFFC0000000LL) == *v11 )
          {
            result = v11[1] + ((v9 >> 9) & 0x1FFFF8);
          }
          else if ( v10 == v3[32] )
          {
            v3[32] = v12;
            v13 = v3[33];
            v3[33] = v11[1];
            *v11 = v10;
            v11[1] = v13;
            result = v13 + ((v9 >> 9) & 0x1FFFF8);
          }
          else
          {
            v14 = v3 + 34;
            for ( i = 1; i != 8; ++i )
            {
              if ( v10 == *v14 )
              {
                v16 = i;
                v17 = &v3[2 * i - 2];
                v18 = &v3[2 * v16];
                v19 = v18[33];
                v18[32] = v17[32];
                v18[33] = v17[33];
                v17[32] = v12;
                v17[33] = v11[1];
                v11[1] = v19;
                *v11 = v10;
                result = ((v9 >> 9) & 0x1FFFF8) + v19;
                goto LABEL_8;
              }
              v14 += 2;
            }
            result = sub_130D370(a1, a2, v3, v9, 1, 0);
          }
        }
LABEL_8:
        v9 += 4096LL;
        result += 8LL;
        *(_QWORD *)(result - 8) = 0xE8000000000000LL;
      }
      while ( v7 >= v9 );
    }
  }
  return result;
}
