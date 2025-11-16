// Function: sub_334DB50
// Address: 0x334db50
//
__int64 __fastcall sub_334DB50(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rcx
  __int64 result; // rax
  unsigned __int64 v6; // rdx
  _BYTE *v7; // rsi
  int v8; // r15d
  int v9; // r14d
  __int64 v10; // r8
  __int64 v11; // r13
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // rdx
  __int64 v15; // rdi
  unsigned int v16; // esi
  __int64 v17; // r10
  _WORD *v19; // rdx
  unsigned int v20; // r11d
  __int64 v21; // [rsp+0h] [rbp-40h]
  __int64 v22[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = *(_DWORD *)(a2 + 36);
  v22[0] = a2;
  if ( v2 )
    BUG();
  v3 = *(unsigned int *)(a2 + 24);
  result = a2;
  if ( (int)v3 < 0 )
    goto LABEL_35;
  LOBYTE(v6) = 0;
  if ( (int)v3 <= 45 )
    v6 = (0x3FF8000FFE42uLL >> v3) & 1;
  if ( (_DWORD)v3 != 324 && !(_BYTE)v6 )
  {
LABEL_35:
    v7 = *(_BYTE **)(a1 + 640);
    if ( v7 == *(_BYTE **)(a1 + 648) )
    {
      sub_334D9C0(a1 + 632, v7, v22);
      result = v22[0];
    }
    else
    {
      if ( v7 )
      {
        *(_QWORD *)v7 = result;
        v7 = *(_BYTE **)(a1 + 640);
      }
      *(_QWORD *)(a1 + 640) = v7 + 8;
    }
    v8 = *(_DWORD *)(result + 64);
    if ( v8 )
    {
      v9 = *(_DWORD *)(result + 64);
      v10 = 0;
      v11 = 40LL * (unsigned int)(v8 - 1);
      v12 = v11 + *(_QWORD *)(result + 40);
      v13 = *(_QWORD *)v12;
LABEL_24:
      v19 = (_WORD *)(*(_QWORD *)(v13 + 48) + 16LL * *(unsigned int *)(v12 + 8));
      if ( *v19 == 262 )
      {
        *(_DWORD *)(v13 + 36) = 0;
        result = sub_334DB50(a1, v13, v19, v3, v10);
      }
      else
      {
        while ( v10 != v13 )
        {
          v14 = *(unsigned int *)(a1 + 680);
          v15 = *(_QWORD *)(a1 + 664);
          if ( (_DWORD)v14 )
          {
            v16 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v3 = v15 + 16LL * v16;
            v17 = *(_QWORD *)v3;
            if ( v13 == *(_QWORD *)v3 )
            {
LABEL_17:
              v14 = v15 + 16 * v14;
              if ( v3 != v14 )
              {
                v14 = *(_QWORD *)(v3 + 8);
                if ( v14 != result )
                  v13 = *(_QWORD *)(v3 + 8);
              }
            }
            else
            {
              v3 = 1;
              while ( v17 != -4096 )
              {
                v20 = v3 + 1;
                v16 = (v14 - 1) & (v3 + v16);
                v3 = v15 + 16LL * v16;
                v17 = *(_QWORD *)v3;
                if ( v13 == *(_QWORD *)v3 )
                  goto LABEL_17;
                v3 = v20;
              }
            }
          }
          if ( (*(_DWORD *)(v13 + 36))-- == 1 )
          {
            v21 = v10;
            result = sub_334DB50(a1, v13, v14, v3, v10);
            v10 = v21;
          }
LABEL_22:
          v11 -= 40;
          --v9;
          if ( v11 == -40 )
            return result;
          result = v22[0];
          v12 = v11 + *(_QWORD *)(v22[0] + 40);
          v13 = *(_QWORD *)v12;
          if ( v8 == v9 )
            goto LABEL_24;
        }
      }
      v10 = v13;
      goto LABEL_22;
    }
  }
  return result;
}
