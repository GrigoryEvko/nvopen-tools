// Function: sub_1CFD950
// Address: 0x1cfd950
//
__int64 __fastcall sub_1CFD950(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rcx
  unsigned __int64 v5; // rdx
  _BYTE *v6; // rsi
  int v7; // r15d
  int v8; // r14d
  __int64 v9; // r8
  __int64 v10; // r13
  __int64 v11; // rdx
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rdi
  unsigned int v15; // esi
  __int64 v16; // r10
  _BYTE *v18; // rdx
  unsigned int v19; // r11d
  __int64 v20; // [rsp+0h] [rbp-40h]
  __int64 v21[7]; // [rsp+8h] [rbp-38h] BYREF

  result = a2;
  v4 = *(unsigned __int16 *)(a2 + 24);
  v21[0] = a2;
  if ( (v4 & 0x8000u) != 0LL )
    goto LABEL_33;
  LOBYTE(v5) = 0;
  if ( (__int16)v4 <= 42 )
    v5 = (0x7FF0007FF22uLL >> v4) & 1;
  if ( (_WORD)v4 != 209 && !(_BYTE)v5 )
  {
LABEL_33:
    v6 = *(_BYTE **)(a1 + 672);
    if ( v6 == *(_BYTE **)(a1 + 680) )
    {
      sub_1CFD7C0(a1 + 664, v6, v21);
      result = v21[0];
    }
    else
    {
      if ( v6 )
      {
        *(_QWORD *)v6 = result;
        v6 = *(_BYTE **)(a1 + 672);
      }
      *(_QWORD *)(a1 + 672) = v6 + 8;
    }
    v7 = *(_DWORD *)(result + 56);
    if ( v7 )
    {
      v8 = *(_DWORD *)(result + 56);
      v9 = 0;
      v10 = 40LL * (unsigned int)(v7 - 1);
      v11 = v10 + *(_QWORD *)(result + 32);
      v12 = *(_QWORD *)v11;
LABEL_23:
      v18 = (_BYTE *)(*(_QWORD *)(v12 + 40) + 16LL * *(unsigned int *)(v11 + 8));
      if ( *v18 == 111 )
      {
        *(_DWORD *)(v12 + 28) = 0;
        result = sub_1CFD950(a1, v12, v18, v4, v9);
      }
      else
      {
        while ( v9 != v12 )
        {
          v13 = *(unsigned int *)(a1 + 712);
          if ( (_DWORD)v13 )
          {
            v14 = *(_QWORD *)(a1 + 696);
            v15 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
            v4 = v14 + 16LL * v15;
            v16 = *(_QWORD *)v4;
            if ( v12 == *(_QWORD *)v4 )
            {
LABEL_16:
              v13 = v14 + 16 * v13;
              if ( v4 != v13 )
              {
                v13 = *(_QWORD *)(v4 + 8);
                if ( v13 != result )
                  v12 = *(_QWORD *)(v4 + 8);
              }
            }
            else
            {
              v4 = 1;
              while ( v16 != -8 )
              {
                v19 = v4 + 1;
                v15 = (v13 - 1) & (v4 + v15);
                v4 = v14 + 16LL * v15;
                v16 = *(_QWORD *)v4;
                if ( v12 == *(_QWORD *)v4 )
                  goto LABEL_16;
                v4 = v19;
              }
            }
          }
          if ( (*(_DWORD *)(v12 + 28))-- == 1 )
          {
            v20 = v9;
            result = sub_1CFD950(a1, v12, v13, v4, v9);
            v9 = v20;
          }
LABEL_21:
          v10 -= 40;
          --v8;
          if ( v10 == -40 )
            return result;
          result = v21[0];
          v11 = v10 + *(_QWORD *)(v21[0] + 32);
          v12 = *(_QWORD *)v11;
          if ( v7 == v8 )
            goto LABEL_23;
        }
      }
      v9 = v12;
      goto LABEL_21;
    }
  }
  return result;
}
