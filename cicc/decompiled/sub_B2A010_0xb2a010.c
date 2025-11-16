// Function: sub_B2A010
// Address: 0xb2a010
//
__int64 *__fastcall sub_B2A010(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v8; // rcx
  __int64 *result; // rax
  unsigned int v10; // ebx
  __int64 v11; // r14
  __int64 *v12; // r13
  __int64 *v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned int v16; // eax
  __int64 v17; // rbx
  _BYTE **v18; // rsi
  __int64 v19; // rbx
  __int64 v20; // rsi
  __int64 *v21; // r14
  __int64 v22; // rdi
  unsigned int v23; // eax
  __int64 v24; // rdx
  __int64 *v25; // rdx
  __int64 *v26; // [rsp+0h] [rbp-A0h]
  __int64 *v27; // [rsp+8h] [rbp-98h]
  unsigned int v28; // [rsp+14h] [rbp-8Ch]
  __int64 *v29; // [rsp+18h] [rbp-88h]
  __int64 *v30; // [rsp+20h] [rbp-80h] BYREF
  int v31; // [rsp+28h] [rbp-78h]
  _BYTE v32[112]; // [rsp+30h] [rbp-70h] BYREF

  if ( a3 )
  {
    v8 = (__int64 *)(unsigned int)(*(_DWORD *)(a3 + 44) + 1);
    result = v8;
  }
  else
  {
    v8 = 0;
    result = 0;
  }
  v10 = *(_DWORD *)(a1 + 56);
  if ( v10 <= (unsigned int)result )
    return result;
  v11 = *(_QWORD *)(a1 + 48);
  v12 = *(__int64 **)(v11 + 8LL * (_QWORD)v8);
  if ( !v12 )
    return result;
  if ( a4 )
  {
    v13 = (__int64 *)(unsigned int)(*(_DWORD *)(a4 + 44) + 1);
    result = v13;
  }
  else
  {
    v13 = 0;
    result = 0;
  }
  if ( v10 <= (unsigned int)result )
    return result;
  result = *(__int64 **)(v11 + 8LL * (_QWORD)v13);
  v29 = result;
  if ( !result )
    return result;
  v14 = sub_B197A0(a1, a3, a4);
  if ( v14 )
  {
    v15 = (unsigned int)(*(_DWORD *)(v14 + 44) + 1);
    v16 = *(_DWORD *)(v14 + 44) + 1;
  }
  else
  {
    v15 = 0;
    v16 = 0;
  }
  if ( v10 <= v16 || v29 != *(__int64 **)(v11 + 8 * v15) )
  {
    *(_BYTE *)(a1 + 136) = 0;
    if ( v12 == (__int64 *)v29[1] )
    {
      v19 = *v29;
      v20 = *v29;
      sub_B1CB80(&v30, *v29, a2);
      v21 = v30;
      v26 = v30;
      v27 = &v30[v31];
      if ( v30 == v27 )
      {
LABEL_33:
        if ( v26 != (__int64 *)v32 )
          _libc_free(v26, v20);
        sub_B1A4E0(a1, *v29);
        v25 = 0;
        if ( *(_DWORD *)(a1 + 56) )
          v25 = **(__int64 ***)(a1 + 48);
        sub_B29620((_QWORD **)a1, a2, v25, (__int64)v29);
        goto LABEL_15;
      }
      v28 = *(_DWORD *)(a1 + 56);
      while ( 1 )
      {
        v24 = *v21;
        if ( *v21 )
        {
          v22 = (unsigned int)(*(_DWORD *)(v24 + 44) + 1);
          v23 = *(_DWORD *)(v24 + 44) + 1;
        }
        else
        {
          v22 = 0;
          v23 = 0;
        }
        if ( v23 < v28 )
        {
          if ( *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v22) )
          {
            v20 = v19;
            if ( v19 != sub_B197A0(a1, v19, v24) )
              break;
          }
        }
        if ( v27 == ++v21 )
          goto LABEL_33;
      }
      if ( v26 != (__int64 *)v32 )
        _libc_free(v26, v19);
    }
    sub_B29130(a1, a2, *v12, *v29);
  }
LABEL_15:
  v17 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  result = sub_B1D3D0(*(__int64 **)a1, v17, a2);
  if ( (__int64 *)v17 != result )
  {
    sub_B28710(&v30, a1, a2);
    v18 = (_BYTE **)&v30;
    if ( !(unsigned __int8)sub_B1B2E0(a1, (__int64)&v30) )
    {
      v18 = (_BYTE **)a2;
      sub_B28E70(a1, a2);
    }
    result = (__int64 *)v32;
    if ( v30 != (__int64 *)v32 )
      return (__int64 *)_libc_free(v30, v18);
  }
  return result;
}
