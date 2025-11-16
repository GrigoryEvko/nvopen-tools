// Function: sub_1421110
// Address: 0x1421110
//
__int64 __fastcall sub_1421110(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 result; // rax
  __int64 v7; // r15
  unsigned int i; // ebx
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned int v14; // edi
  __int64 *v15; // rdx
  __int64 v16; // r10
  __int64 v17; // r14
  __int64 v18; // r9
  __int64 v19; // rdi
  int v20; // esi
  __int64 v21; // rax
  char v22; // r10
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // r9
  __int64 *v27; // r9
  __int64 v28; // rdx
  unsigned __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rsi
  unsigned int v32; // edx
  int v33; // eax
  __int64 v34; // rsi
  __int64 *v35; // rax
  __int64 v36; // rsi
  unsigned __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // r9
  int v41; // edx
  __int64 v42; // rsi
  __int64 v43; // [rsp+10h] [rbp-40h]
  int v45; // [rsp+1Ch] [rbp-34h]

  result = sub_157EBA0(a2);
  if ( result )
  {
    v45 = sub_15F4D60(result);
    result = sub_157EBA0(a2);
    v7 = result;
    if ( v45 )
    {
      v43 = a3;
      for ( i = 0; i != v45; ++i )
      {
        v11 = sub_15F4DF0(v7, i);
        result = *(unsigned int *)(a1 + 80);
        if ( (_DWORD)result )
        {
          v13 = *(_QWORD *)(a1 + 64);
          v14 = (result - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v15 = (__int64 *)(v13 + 16LL * v14);
          v16 = *v15;
          if ( v11 == *v15 )
          {
LABEL_6:
            result = v13 + 16 * result;
            if ( v15 != (__int64 *)result )
            {
              result = v15[1];
              v17 = *(_QWORD *)(result + 8);
              if ( !v17 )
                BUG();
              if ( *(_BYTE *)(v17 - 16) == 23 )
              {
                v18 = v17 - 32;
                v19 = *(_DWORD *)(v17 - 12) & 0xFFFFFFF;
                v20 = *(_DWORD *)(v17 - 12) & 0xFFFFFFF;
                if ( a4 )
                {
                  v21 = 0x17FFFFFFE8LL;
                  v22 = *(_BYTE *)(v17 - 9) & 0x40;
                  if ( (_DWORD)v19 )
                  {
                    v23 = 24LL * *(unsigned int *)(v17 + 44) + 8;
                    v24 = 0;
                    do
                    {
                      v25 = v18 - 24LL * (unsigned int)v19;
                      if ( v22 )
                        v25 = *(_QWORD *)(v17 - 40);
                      if ( a2 == *(_QWORD *)(v25 + v23) )
                      {
                        v21 = 24 * v24;
                        goto LABEL_17;
                      }
                      ++v24;
                      v23 += 8;
                    }
                    while ( (_DWORD)v19 != (_DWORD)v24 );
                    v21 = 0x17FFFFFFE8LL;
                  }
LABEL_17:
                  if ( v22 )
                    v26 = *(_QWORD *)(v17 - 40);
                  else
                    v26 = v18 - 24 * v19;
                  v27 = (__int64 *)(v21 + v26);
                  if ( *v27 )
                  {
                    v28 = v27[1];
                    v29 = v27[2] & 0xFFFFFFFFFFFFFFFCLL;
                    *(_QWORD *)v29 = v28;
                    if ( v28 )
                      *(_QWORD *)(v28 + 16) = *(_QWORD *)(v28 + 16) & 3LL | v29;
                  }
                  result = v43;
                  *v27 = v43;
                  if ( v43 )
                  {
                    v30 = *(_QWORD *)(v43 + 8);
                    v27[1] = v30;
                    if ( v30 )
                      *(_QWORD *)(v30 + 16) = (unsigned __int64)(v27 + 1) | *(_QWORD *)(v30 + 16) & 3LL;
                    v27[2] = (v43 + 8) | v27[2] & 3;
                    result = v43;
                    *(_QWORD *)(v43 + 8) = v27;
                  }
                }
                else
                {
                  if ( (_DWORD)v19 == *(_DWORD *)(v17 + 44) )
                  {
                    v42 = ((unsigned int)v19 >> 1) + v20;
                    if ( (unsigned int)v42 < 2 )
                      v42 = 2;
                    *(_DWORD *)(v17 + 44) = v42;
                    sub_16488D0(v17 - 32, v42, 1, v10, v12, v18);
                    v18 = v17 - 32;
                    v20 = *(_DWORD *)(v17 - 12) & 0xFFFFFFF;
                  }
                  v31 = (v20 + 1) & 0xFFFFFFF;
                  v32 = v31 - 1;
                  v33 = v31 | *(_DWORD *)(v17 - 12) & 0xF0000000;
                  *(_DWORD *)(v17 - 12) = v33;
                  if ( (v33 & 0x40000000) != 0 )
                    v34 = *(_QWORD *)(v17 - 40);
                  else
                    v34 = v18 - 24 * v31;
                  v35 = (__int64 *)(v34 + 24LL * v32);
                  if ( *v35 )
                  {
                    v36 = v35[1];
                    v37 = v35[2] & 0xFFFFFFFFFFFFFFFCLL;
                    *(_QWORD *)v37 = v36;
                    if ( v36 )
                      *(_QWORD *)(v36 + 16) = *(_QWORD *)(v36 + 16) & 3LL | v37;
                  }
                  *v35 = v43;
                  if ( v43 )
                  {
                    v38 = *(_QWORD *)(v43 + 8);
                    v35[1] = v38;
                    if ( v38 )
                      *(_QWORD *)(v38 + 16) = (unsigned __int64)(v35 + 1) | *(_QWORD *)(v38 + 16) & 3LL;
                    v35[2] = (v43 + 8) | v35[2] & 3;
                    *(_QWORD *)(v43 + 8) = v35;
                  }
                  v39 = *(_DWORD *)(v17 - 12) & 0xFFFFFFF;
                  if ( (*(_BYTE *)(v17 - 9) & 0x40) != 0 )
                    v40 = *(_QWORD *)(v17 - 40);
                  else
                    v40 = v18 - 24 * v39;
                  result = 8LL * (unsigned int)(v39 - 1) + 24LL * *(unsigned int *)(v17 + 44);
                  *(_QWORD *)(v40 + result + 8) = a2;
                }
              }
            }
          }
          else
          {
            v41 = 1;
            while ( v16 != -8 )
            {
              v10 = (unsigned int)(v41 + 1);
              v14 = (result - 1) & (v41 + v14);
              v15 = (__int64 *)(v13 + 16LL * v14);
              v16 = *v15;
              if ( v11 == *v15 )
                goto LABEL_6;
              v41 = v10;
            }
          }
        }
      }
    }
  }
  return result;
}
