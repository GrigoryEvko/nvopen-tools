// Function: sub_103BCD0
// Address: 0x103bcd0
//
__int64 __fastcall sub_103BCD0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 result; // rax
  __int64 v6; // r15
  unsigned int v9; // r14d
  int v10; // eax
  unsigned int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rcx
  unsigned int v18; // esi
  __int64 *v19; // rdx
  __int64 v20; // r9
  __int64 v21; // rcx
  __int64 v22; // rsi
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 v27; // rdi
  int v28; // edx
  int v29; // r8d
  unsigned int v30; // esi
  __int64 v31; // [rsp+8h] [rbp-48h]
  __int64 v32; // [rsp+10h] [rbp-40h]
  int v34; // [rsp+1Ch] [rbp-34h]

  result = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( result != a2 + 48 )
  {
    if ( !result )
      BUG();
    v6 = result - 24;
    result = (unsigned int)*(unsigned __int8 *)(result - 24) - 30;
    if ( (unsigned int)result <= 0xA )
    {
      result = sub_B46E30(v6);
      v34 = result;
      if ( (_DWORD)result )
      {
        v9 = 0;
        v32 = a3 + 16;
        do
        {
          v15 = sub_B46EC0(v6, v9);
          v16 = *(_QWORD *)(a1 + 72);
          v17 = v15;
          result = *(unsigned int *)(a1 + 88);
          if ( (_DWORD)result )
          {
            v18 = (result - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
            v19 = (__int64 *)(v16 + 16LL * v18);
            v20 = *v19;
            if ( v17 == *v19 )
            {
LABEL_18:
              result = v16 + 16 * result;
              if ( v19 != (__int64 *)result )
              {
                result = v19[1];
                v21 = *(_QWORD *)(result + 8);
                if ( !v21 )
                  BUG();
                if ( *(_BYTE *)(v21 - 32) == 28 )
                {
                  v22 = *(_DWORD *)(v21 - 28) & 0x7FFFFFF;
                  result = v22;
                  if ( a4 )
                  {
                    if ( (_DWORD)v22 )
                    {
                      v23 = 8 * v22;
                      v24 = 0;
                      do
                      {
                        v25 = *(_QWORD *)(v21 - 40);
                        result = v25 + 32LL * *(unsigned int *)(v21 + 44);
                        if ( a2 == *(_QWORD *)(result + v24) )
                        {
                          result = v25 + 4 * v24;
                          if ( *(_QWORD *)result )
                          {
                            v26 = *(_QWORD *)(result + 8);
                            **(_QWORD **)(result + 16) = v26;
                            if ( v26 )
                              *(_QWORD *)(v26 + 16) = *(_QWORD *)(result + 16);
                          }
                          *(_QWORD *)result = a3;
                          if ( a3 )
                          {
                            v27 = *(_QWORD *)(a3 + 16);
                            *(_QWORD *)(result + 8) = v27;
                            if ( v27 )
                              *(_QWORD *)(v27 + 16) = result + 8;
                            *(_QWORD *)(result + 16) = v32;
                            *(_QWORD *)(a3 + 16) = result;
                          }
                        }
                        v24 += 8;
                      }
                      while ( v24 != v23 );
                    }
                  }
                  else
                  {
                    if ( (_DWORD)v22 == *(_DWORD *)(v21 + 44) )
                    {
                      v31 = v21;
                      v30 = v22 + ((unsigned int)v22 >> 1);
                      if ( v30 < 2 )
                        v30 = 2;
                      *(_DWORD *)(v21 + 44) = v30;
                      sub_BD2A80(v21 - 32, v30, 1);
                      v21 = v31;
                      LODWORD(result) = *(_DWORD *)(v31 - 28) & 0x7FFFFFF;
                    }
                    v10 = (result + 1) & 0x7FFFFFF;
                    v11 = v10 | *(_DWORD *)(v21 - 28) & 0xF8000000;
                    v12 = *(_QWORD *)(v21 - 40) + 32LL * (unsigned int)(v10 - 1);
                    *(_DWORD *)(v21 - 28) = v11;
                    if ( *(_QWORD *)v12 )
                    {
                      v13 = *(_QWORD *)(v12 + 8);
                      **(_QWORD **)(v12 + 16) = v13;
                      if ( v13 )
                        *(_QWORD *)(v13 + 16) = *(_QWORD *)(v12 + 16);
                    }
                    *(_QWORD *)v12 = a3;
                    if ( a3 )
                    {
                      v14 = *(_QWORD *)(a3 + 16);
                      *(_QWORD *)(v12 + 8) = v14;
                      if ( v14 )
                        *(_QWORD *)(v14 + 16) = v12 + 8;
                      *(_QWORD *)(v12 + 16) = v32;
                      *(_QWORD *)(a3 + 16) = v12;
                    }
                    result = *(_QWORD *)(v21 - 40)
                           + 32LL * *(unsigned int *)(v21 + 44)
                           + 8LL * ((*(_DWORD *)(v21 - 28) & 0x7FFFFFFu) - 1);
                    *(_QWORD *)result = a2;
                  }
                }
              }
            }
            else
            {
              v28 = 1;
              while ( v20 != -4096 )
              {
                v29 = v28 + 1;
                v18 = (result - 1) & (v28 + v18);
                v19 = (__int64 *)(v16 + 16LL * v18);
                v20 = *v19;
                if ( v17 == *v19 )
                  goto LABEL_18;
                v28 = v29;
              }
            }
          }
          ++v9;
        }
        while ( v34 != v9 );
      }
    }
  }
  return result;
}
