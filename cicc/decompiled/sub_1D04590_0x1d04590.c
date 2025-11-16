// Function: sub_1D04590
// Address: 0x1d04590
//
unsigned int *__fastcall sub_1D04590(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, unsigned int *a5, _QWORD *a6)
{
  __int64 v6; // rcx
  __int64 v7; // rdx
  unsigned __int16 v8; // r12
  unsigned int v9; // edi
  __int16 v10; // r14
  _WORD *v11; // r10
  unsigned __int16 v12; // si
  _WORD *v13; // rdi
  unsigned __int16 v14; // r14
  _WORD *v15; // r10
  _WORD *v16; // r13
  unsigned __int16 *v17; // rax
  unsigned __int16 v18; // r15
  unsigned __int16 *v19; // rbx
  unsigned int *result; // rax
  unsigned __int16 *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  int v24; // r10d
  __int64 v25; // rax
  __int64 v26; // rax
  __int16 v27; // ax
  _QWORD *v31; // [rsp+28h] [rbp-58h]
  unsigned int *v32; // [rsp+30h] [rbp-50h]
  unsigned int v33[13]; // [rsp+4Ch] [rbp-34h] BYREF

  if ( !a6 )
    BUG();
  v6 = a6[1];
  v7 = a6[7];
  v8 = 0;
  v9 = *(_DWORD *)(v6 + 24LL * a2 + 16);
  v10 = a2 * (v9 & 0xF);
  v11 = (_WORD *)(v7 + 2LL * (v9 >> 4));
  v12 = 0;
  v13 = v11 + 1;
  v14 = *v11 + v10;
LABEL_3:
  v15 = v13;
  while ( 1 )
  {
    v16 = v15;
    if ( !v15 )
    {
      v18 = v12;
      v19 = 0;
      goto LABEL_7;
    }
    v17 = (unsigned __int16 *)(a6[6] + 4LL * v14);
    v18 = *v17;
    v8 = v17[1];
    if ( *v17 )
      break;
LABEL_22:
    v27 = *v15;
    v13 = 0;
    ++v15;
    v14 += v27;
    if ( !v27 )
      goto LABEL_3;
  }
  while ( 1 )
  {
    v19 = (unsigned __int16 *)(v7 + 2LL * *(unsigned int *)(v6 + 24LL * v18 + 8));
    if ( v19 )
      break;
    if ( !v8 )
    {
      v12 = v18;
      goto LABEL_22;
    }
    v18 = v8;
    v8 = 0;
  }
LABEL_7:
  result = v33;
  while ( v16 )
  {
    v22 = *(_QWORD *)(a3 + 8LL * v18);
    if ( v22 != a1 )
    {
      if ( v22 )
      {
        v31 = a6;
        v32 = a5;
        v33[0] = v18;
        v23 = sub_1D041C0(a4, v33, v7, a3, (int)a5);
        v24 = v18;
        a5 = v32;
        a6 = v31;
        if ( BYTE4(v23) )
        {
          v26 = v32[2];
          if ( (unsigned int)v26 >= v32[3] )
          {
            sub_16CD150((__int64)v32, v32 + 4, 0, 4, (int)v32, (int)v31);
            a5 = v32;
            a6 = v31;
            v24 = v18;
            v26 = v32[2];
          }
          v7 = *(_QWORD *)a5;
          *(_DWORD *)(*(_QWORD *)a5 + 4 * v26) = v24;
          ++a5[2];
        }
      }
    }
    result = (unsigned int *)*v19++;
    v18 += (unsigned __int16)result;
    if ( !(_WORD)result )
    {
      if ( v8 )
      {
        v25 = v8;
        v18 = v8;
        v8 = 0;
        v7 = *(unsigned int *)(a6[1] + 24 * v25 + 8);
        result = (unsigned int *)a6[7];
        v19 = (unsigned __int16 *)result + v7;
      }
      else
      {
        v14 += *v16;
        if ( !*v16 )
          return result;
        ++v16;
        v21 = (unsigned __int16 *)(a6[6] + 4LL * v14);
        v18 = *v21;
        v8 = v21[1];
        v7 = *(unsigned int *)(a6[1] + 24LL * *v21 + 8);
        result = (unsigned int *)a6[7];
        v19 = (unsigned __int16 *)result + v7;
      }
    }
  }
  return result;
}
