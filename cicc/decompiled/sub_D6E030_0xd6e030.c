// Function: sub_D6E030
// Address: 0xd6e030
//
__int64 __fastcall sub_D6E030(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v8; // r13
  unsigned int v9; // r15d
  __int64 v10; // rdx
  int v11; // edi
  __int64 v12; // r9
  int v13; // edi
  unsigned int v14; // esi
  __int64 *v15; // rax
  __int64 v16; // r11
  int v17; // edx
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rsi
  int v21; // edx
  __int64 v22; // r9
  __int64 v23; // rdx
  unsigned int v24; // r10d
  int v25; // [rsp+Ch] [rbp-34h]

  sub_D6DC00(a1, a2, a3, a4);
  result = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( result != a2 + 48 )
  {
    if ( !result )
      BUG();
    v8 = result - 24;
    result = (unsigned int)*(unsigned __int8 *)(result - 24) - 30;
    if ( (unsigned int)result <= 0xA )
    {
      result = sub_B46E30(v8);
      v25 = result;
      if ( (_DWORD)result )
      {
        v9 = 0;
        do
        {
          v10 = sub_B46EC0(v8, v9);
          result = *(_QWORD *)a1;
          v11 = *(_DWORD *)(*(_QWORD *)a1 + 56LL);
          v12 = *(_QWORD *)(*(_QWORD *)a1 + 40LL);
          if ( v11 )
          {
            v13 = v11 - 1;
            v14 = v13 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
            v15 = (__int64 *)(v12 + 16LL * v14);
            v16 = *v15;
            if ( v10 == *v15 )
            {
LABEL_8:
              result = v15[1];
              if ( result )
              {
                v17 = *(_DWORD *)(result + 4);
                v18 = *(_QWORD *)(result - 8);
                v19 = 32LL * *(unsigned int *)(result + 76);
                v20 = v19;
                v21 = v17 & 0x7FFFFFF;
                if ( v21 )
                {
                  v22 = v18 + v19 + 8;
                  result = v18 + v19;
                  v23 = v22 + 8LL * (unsigned int)(v21 - 1);
                  while ( a2 != *(_QWORD *)result )
                  {
                    result += 8;
                    if ( result == v23 )
                      goto LABEL_16;
                  }
                }
                else
                {
LABEL_16:
                  result = v18 + v20 + 0x7FFFFFFF8LL;
                }
                *(_QWORD *)result = a3;
              }
            }
            else
            {
              result = 1;
              while ( v16 != -4096 )
              {
                v24 = result + 1;
                v14 = v13 & (result + v14);
                v15 = (__int64 *)(v12 + 16LL * v14);
                v16 = *v15;
                if ( v10 == *v15 )
                  goto LABEL_8;
                result = v24;
              }
            }
          }
          ++v9;
        }
        while ( v25 != v9 );
      }
    }
  }
  return result;
}
