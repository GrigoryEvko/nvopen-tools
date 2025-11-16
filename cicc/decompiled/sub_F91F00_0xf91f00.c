// Function: sub_F91F00
// Address: 0xf91f00
//
__int64 __fastcall sub_F91F00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // r13
  __int64 v11; // r8
  __int64 v12; // rax
  unsigned int v13; // r10d
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // r9
  int v17; // edx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rcx
  int v22; // esi
  unsigned int v23; // edx
  __int64 v24; // rdi
  __int64 v25; // rbx
  __int64 v26; // rax
  unsigned int v27; // edi
  __int64 v28; // rsi
  unsigned int v29; // edx
  __int64 v30; // rax
  __int64 v31; // r12
  int v32; // edx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rdx
  unsigned int v36; // esi
  unsigned int v37; // r8d
  __int64 v39; // [rsp+8h] [rbp-38h]

  result = sub_AA5930(a1);
  v9 = v8;
  v10 = result;
  while ( v9 != v10 )
  {
    v11 = *(_QWORD *)(v10 - 8);
    v12 = 0x1FFFFFFFE0LL;
    v13 = *(_DWORD *)(v10 + 72);
    v14 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
    if ( v14 )
    {
      v15 = 0;
      do
      {
        if ( a3 == *(_QWORD *)(v11 + 32LL * v13 + 8 * v15) )
        {
          v12 = 32 * v15;
          goto LABEL_7;
        }
        ++v15;
      }
      while ( v14 != (_DWORD)v15 );
      v16 = *(_QWORD *)(v11 + 0x1FFFFFFFE0LL);
      if ( v14 != v13 )
        goto LABEL_8;
    }
    else
    {
LABEL_7:
      v16 = *(_QWORD *)(v11 + v12);
      if ( v14 != v13 )
        goto LABEL_8;
    }
    v39 = v16;
    sub_B48D90(v10);
    v11 = *(_QWORD *)(v10 - 8);
    v16 = v39;
    v14 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
LABEL_8:
    v17 = (v14 + 1) & 0x7FFFFFF;
    *(_DWORD *)(v10 + 4) = v17 | *(_DWORD *)(v10 + 4) & 0xF8000000;
    v18 = v11 + 32LL * (unsigned int)(v17 - 1);
    if ( *(_QWORD *)v18 )
    {
      v19 = *(_QWORD *)(v18 + 8);
      **(_QWORD **)(v18 + 16) = v19;
      if ( v19 )
        *(_QWORD *)(v19 + 16) = *(_QWORD *)(v18 + 16);
    }
    *(_QWORD *)v18 = v16;
    if ( v16 )
    {
      v20 = *(_QWORD *)(v16 + 16);
      *(_QWORD *)(v18 + 8) = v20;
      if ( v20 )
        *(_QWORD *)(v20 + 16) = v18 + 8;
      *(_QWORD *)(v18 + 16) = v16 + 16;
      *(_QWORD *)(v16 + 16) = v18;
    }
    *(_QWORD *)(*(_QWORD *)(v10 - 8)
              + 32LL * *(unsigned int *)(v10 + 72)
              + 8LL * ((*(_DWORD *)(v10 + 4) & 0x7FFFFFFu) - 1)) = a2;
    result = *(_QWORD *)(v10 + 32);
    if ( !result )
      BUG();
    v10 = 0;
    if ( *(_BYTE *)(result - 24) == 84 )
      v10 = result - 24;
  }
  if ( a4 )
  {
    v21 = *(_QWORD *)(*(_QWORD *)a4 + 40LL);
    result = *(unsigned int *)(*(_QWORD *)a4 + 56LL);
    if ( (_DWORD)result )
    {
      v22 = result - 1;
      v23 = (result - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      result = v21 + 16LL * v23;
      v24 = *(_QWORD *)result;
      if ( a1 == *(_QWORD *)result )
      {
LABEL_22:
        v25 = *(_QWORD *)(result + 8);
        if ( v25 )
        {
          v26 = 0x1FFFFFFFE0LL;
          v27 = *(_DWORD *)(v25 + 76);
          v28 = *(_QWORD *)(v25 - 8);
          v29 = *(_DWORD *)(v25 + 4) & 0x7FFFFFF;
          if ( v29 )
          {
            v30 = 0;
            do
            {
              if ( a3 == *(_QWORD *)(v28 + 32LL * v27 + 8 * v30) )
              {
                v26 = 32 * v30;
                goto LABEL_28;
              }
              ++v30;
            }
            while ( v29 != (_DWORD)v30 );
            v26 = 0x1FFFFFFFE0LL;
          }
LABEL_28:
          v31 = *(_QWORD *)(v28 + v26);
          if ( v29 == v27 )
          {
            v36 = v29 + (v29 >> 1);
            if ( v36 < 2 )
              v36 = 2;
            *(_DWORD *)(v25 + 76) = v36;
            sub_BD2A80(v25, v36, 1);
            v28 = *(_QWORD *)(v25 - 8);
            v29 = *(_DWORD *)(v25 + 4) & 0x7FFFFFF;
          }
          v32 = (v29 + 1) & 0x7FFFFFF;
          *(_DWORD *)(v25 + 4) = v32 | *(_DWORD *)(v25 + 4) & 0xF8000000;
          v33 = v28 + 32LL * (unsigned int)(v32 - 1);
          if ( *(_QWORD *)v33 )
          {
            v34 = *(_QWORD *)(v33 + 8);
            **(_QWORD **)(v33 + 16) = v34;
            if ( v34 )
              *(_QWORD *)(v34 + 16) = *(_QWORD *)(v33 + 16);
          }
          *(_QWORD *)v33 = v31;
          if ( v31 )
          {
            v35 = *(_QWORD *)(v31 + 16);
            *(_QWORD *)(v33 + 8) = v35;
            if ( v35 )
              *(_QWORD *)(v35 + 16) = v33 + 8;
            *(_QWORD *)(v33 + 16) = v31 + 16;
            *(_QWORD *)(v31 + 16) = v33;
          }
          result = *(_QWORD *)(v25 - 8)
                 + 32LL * *(unsigned int *)(v25 + 76)
                 + 8LL * ((*(_DWORD *)(v25 + 4) & 0x7FFFFFFu) - 1);
          *(_QWORD *)result = a2;
        }
      }
      else
      {
        result = 1;
        while ( v24 != -4096 )
        {
          v37 = result + 1;
          v23 = v22 & (result + v23);
          result = v21 + 16LL * v23;
          v24 = *(_QWORD *)result;
          if ( a1 == *(_QWORD *)result )
            goto LABEL_22;
          result = v37;
        }
      }
    }
  }
  return result;
}
