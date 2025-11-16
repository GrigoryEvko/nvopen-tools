// Function: sub_13FDB30
// Address: 0x13fdb30
//
unsigned __int64 __fastcall sub_13FDB30(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdi
  unsigned __int64 result; // rax
  int v4; // ecx
  unsigned int v6; // edx
  __int64 v7; // rsi
  _QWORD *v8; // r8
  _QWORD *v9; // rbx
  _BYTE *v10; // rsi
  __int64 v11; // rsi
  _QWORD *v12; // rax
  _QWORD *v13; // rdi
  unsigned int v14; // r8d
  _QWORD *v15; // rcx
  __int64 v16; // rax
  _BYTE *v17; // rsi
  __int64 *v18; // rdx
  __int64 *v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rsi
  __int64 *v23; // rdx
  __int64 v24; // rcx
  _QWORD *v25; // rsi
  _BYTE *v26; // rsi
  unsigned int v27; // r9d
  _QWORD *v28; // [rsp-38h] [rbp-38h] BYREF
  _QWORD v29[6]; // [rsp-30h] [rbp-30h] BYREF

  v2 = *a1;
  result = *(unsigned int *)(v2 + 24);
  if ( !(_DWORD)result )
    return result;
  v4 = result - 1;
  v6 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = *(_QWORD *)(v2 + 8);
  result = v7 + 16LL * v6;
  v8 = *(_QWORD **)result;
  if ( a2 == *(_QWORD *)result )
  {
LABEL_3:
    v9 = *(_QWORD **)(result + 8);
    v28 = v9;
    if ( !v9 )
      return result;
    if ( a2 == *(_QWORD *)v9[4] )
    {
      v16 = *v9;
      if ( *v9 )
      {
        v17 = *(_BYTE **)(v16 + 16);
        if ( v17 == *(_BYTE **)(v16 + 24) )
        {
          sub_13FD960(v16 + 8, v17, &v28);
          v9 = v28;
        }
        else
        {
          if ( v17 )
          {
            *(_QWORD *)v17 = v9;
            v17 = *(_BYTE **)(v16 + 16);
          }
          *(_QWORD *)(v16 + 16) = v17 + 8;
        }
      }
      else
      {
        v29[0] = v9;
        v26 = *(_BYTE **)(v2 + 40);
        if ( v26 == *(_BYTE **)(v2 + 48) )
        {
          sub_13FD960(v2 + 32, v26, v29);
          v9 = v28;
        }
        else
        {
          if ( v26 )
          {
            *(_QWORD *)v26 = v9;
            v26 = *(_BYTE **)(v2 + 40);
            v9 = v28;
          }
          *(_QWORD *)(v2 + 40) = v26 + 8;
        }
      }
      v18 = (__int64 *)v9[5];
      v19 = (__int64 *)(v9[4] + 8LL);
      if ( v18 != v19 )
      {
        v20 = v18 - 1;
        if ( v19 < v20 )
        {
          do
          {
            v21 = *v19;
            v22 = *v20;
            ++v19;
            --v20;
            *(v19 - 1) = v22;
            v20[1] = v21;
          }
          while ( v19 < v20 );
          v9 = v28;
        }
      }
      result = v9[2];
      v23 = (__int64 *)v9[1];
      if ( v23 != (__int64 *)result )
      {
        result -= 8LL;
        if ( (unsigned __int64)v23 < result )
        {
          do
          {
            v24 = *v23;
            v25 = *(_QWORD **)result;
            ++v23;
            result -= 8LL;
            *(v23 - 1) = (__int64)v25;
            *(_QWORD *)(result + 8) = v24;
          }
          while ( (unsigned __int64)v23 < result );
          v9 = v28;
        }
      }
      v9 = (_QWORD *)*v9;
      v28 = v9;
      if ( !v9 )
        return result;
    }
    while ( 1 )
    {
LABEL_8:
      v29[0] = a2;
      v10 = (_BYTE *)v9[5];
      if ( v10 == (_BYTE *)v9[6] )
      {
        sub_1292090((__int64)(v9 + 4), v10, v29);
        v11 = v29[0];
      }
      else
      {
        if ( v10 )
        {
          *(_QWORD *)v10 = a2;
          v10 = (_BYTE *)v9[5];
        }
        v9[5] = v10 + 8;
        v11 = a2;
      }
      v12 = (_QWORD *)v9[8];
      if ( (_QWORD *)v9[9] != v12 )
        goto LABEL_6;
      v13 = &v12[*((unsigned int *)v9 + 21)];
      v14 = *((_DWORD *)v9 + 21);
      if ( v12 != v13 )
      {
        v15 = 0;
        while ( *v12 != v11 )
        {
          if ( *v12 == -2 )
            v15 = v12;
          if ( v13 == ++v12 )
          {
            if ( !v15 )
              goto LABEL_23;
            *v15 = v11;
            result = (unsigned __int64)v28;
            --*((_DWORD *)v9 + 22);
            ++v9[7];
            v28 = *(_QWORD **)result;
            v9 = v28;
            if ( v28 )
              goto LABEL_8;
            return result;
          }
        }
        goto LABEL_7;
      }
LABEL_23:
      if ( v14 < *((_DWORD *)v9 + 20) )
      {
        *((_DWORD *)v9 + 21) = v14 + 1;
        *v13 = v11;
        ++v9[7];
      }
      else
      {
LABEL_6:
        sub_16CCBA0(v9 + 7, v11);
      }
LABEL_7:
      result = (unsigned __int64)v28;
      v9 = (_QWORD *)*v28;
      v28 = v9;
      if ( !v9 )
        return result;
    }
  }
  result = 1;
  while ( v8 != (_QWORD *)-8LL )
  {
    v27 = result + 1;
    v6 = v4 & (result + v6);
    result = v7 + 16LL * v6;
    v8 = *(_QWORD **)result;
    if ( a2 == *(_QWORD *)result )
      goto LABEL_3;
    result = v27;
  }
  return result;
}
