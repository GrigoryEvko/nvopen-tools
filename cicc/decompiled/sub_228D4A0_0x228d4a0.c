// Function: sub_228D4A0
// Address: 0x228d4a0
//
__int64 __fastcall sub_228D4A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r11
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 result; // rax
  int v8; // r15d
  unsigned int v9; // r8d
  __int64 *v10; // rcx
  __int64 v11; // r13
  __int64 *v12; // rax
  _QWORD *v13; // rdx
  unsigned int v14; // r9d
  __int64 *v15; // rsi
  __int64 v16; // r14
  __int64 *v17; // rdx
  _QWORD **v18; // rdx
  int v19; // r10d
  _QWORD *v20; // r10
  _QWORD *v21; // rcx
  _QWORD *v22; // rsi
  __int64 v23; // rdx
  unsigned int v24; // esi
  int v25; // eax
  __int64 v26; // r10
  int v27; // edx
  int v28; // ecx
  int v29; // esi
  int v30; // r9d
  int v31; // r8d
  int v32; // r10d
  __int64 v33; // rdx
  int v34; // [rsp+0h] [rbp-30h]
  unsigned int v35; // [rsp+4h] [rbp-2Ch]
  int v36; // [rsp+4h] [rbp-2Ch]

  v3 = *(_QWORD *)(a1 + 16);
  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)(a3 + 40);
  v6 = *(_QWORD *)(v3 + 8);
  result = *(unsigned int *)(v3 + 24);
  if ( (_DWORD)result )
  {
    v8 = result - 1;
    v9 = (result - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    v12 = v10;
    if ( v4 == *v10 )
    {
LABEL_3:
      result = v12[1];
      if ( result )
      {
        v13 = *(_QWORD **)result;
        for ( result = 1; v13; result = (unsigned int)(result + 1) )
          v13 = (_QWORD *)*v13;
      }
    }
    else
    {
      v23 = *v10;
      v24 = v9;
      v25 = 1;
      while ( v23 != -4096 )
      {
        v30 = v25 + 1;
        v24 = v8 & (v25 + v24);
        v12 = (__int64 *)(v6 + 16LL * v24);
        v23 = *v12;
        if ( v4 == *v12 )
          goto LABEL_3;
        v25 = v30;
      }
      result = 0;
    }
    v14 = v8 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v15 = (__int64 *)(v6 + 16LL * v14);
    v16 = *v15;
    v17 = v15;
    if ( v5 == *v15 )
    {
LABEL_7:
      v18 = (_QWORD **)v17[1];
      v19 = result;
      if ( v18 )
      {
        v20 = *v18;
        LODWORD(v18) = 1;
        if ( v20 )
        {
          do
          {
            v20 = (_QWORD *)*v20;
            LODWORD(v18) = (_DWORD)v18 + 1;
          }
          while ( v20 );
          v19 = (_DWORD)v18 + result;
        }
        else
        {
          v19 = result + 1;
        }
      }
    }
    else
    {
      v35 = v8 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v26 = *v15;
      v27 = 1;
      while ( v26 != -4096 )
      {
        v32 = v27 + 1;
        v33 = v8 & (v35 + v27);
        v34 = v32;
        v35 = v33;
        v17 = (__int64 *)(v6 + 16 * v33);
        v26 = *v17;
        if ( v5 == *v17 )
          goto LABEL_7;
        v27 = v34;
      }
      v19 = result;
      LODWORD(v18) = 0;
    }
    if ( v4 == v11 )
    {
LABEL_12:
      v21 = (_QWORD *)v10[1];
    }
    else
    {
      v28 = 1;
      while ( v11 != -4096 )
      {
        v9 = v8 & (v28 + v9);
        v36 = v28 + 1;
        v10 = (__int64 *)(v6 + 16LL * v9);
        v11 = *v10;
        if ( v4 == *v10 )
          goto LABEL_12;
        v28 = v36;
      }
      v21 = 0;
    }
    if ( v5 == v16 )
    {
LABEL_14:
      v22 = (_QWORD *)v15[1];
    }
    else
    {
      v29 = 1;
      while ( v16 != -4096 )
      {
        v31 = v29 + 1;
        v14 = v8 & (v29 + v14);
        v15 = (__int64 *)(v6 + 16LL * v14);
        v16 = *v15;
        if ( v5 == *v15 )
          goto LABEL_14;
        v29 = v31;
      }
      v22 = 0;
    }
    *(_DWORD *)(a1 + 36) = result;
    *(_DWORD *)(a1 + 40) = v19;
    if ( (unsigned int)v18 >= (unsigned int)result )
    {
      if ( (unsigned int)v18 > (unsigned int)result )
      {
        do
        {
          LODWORD(v18) = (_DWORD)v18 - 1;
          v22 = (_QWORD *)*v22;
        }
        while ( (_DWORD)v18 != (_DWORD)result );
      }
    }
    else
    {
      do
      {
        result = (unsigned int)(result - 1);
        v21 = (_QWORD *)*v21;
      }
      while ( (_DWORD)result != (_DWORD)v18 );
    }
    for ( ; v21 != v22; result = (unsigned int)(result - 1) )
    {
      v21 = (_QWORD *)*v21;
      v22 = (_QWORD *)*v22;
    }
  }
  else
  {
    *(_DWORD *)(a1 + 36) = 0;
    v19 = 0;
  }
  *(_DWORD *)(a1 + 32) = result;
  *(_DWORD *)(a1 + 40) = v19 - result;
  return result;
}
