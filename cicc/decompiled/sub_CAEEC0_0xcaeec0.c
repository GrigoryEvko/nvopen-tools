// Function: sub_CAEEC0
// Address: 0xcaeec0
//
__int64 __fastcall sub_CAEEC0(__int64 a1, unsigned __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // rdi
  __int64 v8; // rax
  unsigned int v9; // edx
  __m128i v10; // xmm0
  _BYTE *v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  const char *v16; // rax
  __int64 v17; // rdi
  unsigned __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  unsigned int v22; // [rsp+0h] [rbp-A0h] BYREF
  __m128i v23; // [rsp+8h] [rbp-98h]
  _QWORD *v24; // [rsp+18h] [rbp-88h] BYREF
  _QWORD v25[3]; // [rsp+28h] [rbp-78h] BYREF
  _QWORD v26[3]; // [rsp+40h] [rbp-60h] BYREF
  _QWORD *v27; // [rsp+58h] [rbp-48h]
  char v28; // [rsp+60h] [rbp-40h]
  char v29; // [rsp+61h] [rbp-3Fh]
  _QWORD v30[7]; // [rsp+68h] [rbp-38h] BYREF

  result = sub_CA94D0(a1);
  if ( (_BYTE)result )
  {
    *(_BYTE *)(a1 + 77) = 1;
    *(_QWORD *)(a1 + 80) = 0;
    return result;
  }
  v7 = *(_QWORD *)(a1 + 80);
  if ( v7 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
  v8 = sub_CAD7B0(a1, a2, v4, v5, v6);
  v9 = *(_DWORD *)v8;
  v10 = _mm_loadu_si128((const __m128i *)(v8 + 8));
  v24 = v25;
  v11 = *(_BYTE **)(v8 + 24);
  v22 = v9;
  v12 = *(_QWORD *)(v8 + 32);
  v23 = v10;
  sub_CA64F0((__int64 *)&v24, v11, (__int64)&v11[v12]);
  result = *(unsigned int *)(a1 + 72);
  if ( (_DWORD)result )
  {
    if ( (_DWORD)result != 2 )
    {
      if ( (_DWORD)result == 1 )
      {
        result = v22;
        switch ( v22 )
        {
          case 0u:
            goto LABEL_10;
          case 2u:
          case 5u:
          case 6u:
            v29 = 1;
            v16 = "Could not find closing ]!";
            goto LABEL_9;
          case 0xBu:
            sub_CAD6B0((__int64)v26, a1, v13, v14, v15);
            if ( v27 != v30 )
              j_j___libc_free_0(v27, v30[0] + 1LL);
            *(_BYTE *)(a1 + 78) = 1;
            result = sub_CAEEC0(a1);
            v17 = (__int64)v24;
            if ( v24 != v25 )
              return j_j___libc_free_0(v17, v25[0] + 1LL);
            return result;
          case 0xDu:
            goto LABEL_18;
          default:
            if ( !*(_BYTE *)(a1 + 78) )
            {
              v29 = 1;
              v16 = "Expected , between entries!";
              goto LABEL_9;
            }
            result = sub_CAE810(a1, (unsigned __int64)v11, v13, v14, v15);
            *(_QWORD *)(a1 + 80) = result;
            if ( !result )
              *(_BYTE *)(a1 + 77) = 1;
            *(_BYTE *)(a1 + 78) = 0;
            break;
        }
      }
      goto LABEL_11;
    }
    if ( v22 != 7 )
      goto LABEL_10;
  }
  else
  {
    result = v22;
    if ( v22 != 7 )
    {
      if ( v22 == 8 )
      {
LABEL_18:
        sub_CAD6B0((__int64)v26, a1, v13, v14, v15);
        result = (__int64)v30;
        if ( v27 != v30 )
          result = j_j___libc_free_0(v27, v30[0] + 1LL);
      }
      else if ( v22 )
      {
        v29 = 1;
        v16 = "Unexpected token. Expected Block Entry or Block End.";
LABEL_9:
        v26[0] = v16;
        v28 = 3;
        result = (__int64)sub_CA8D00(a1, (__int64)v26, (__int64)&v22, v14, v15);
      }
      goto LABEL_10;
    }
  }
  v18 = a1;
  sub_CAD6B0((__int64)v26, a1, v13, v14, v15);
  if ( v27 != v30 )
  {
    v18 = v30[0] + 1LL;
    j_j___libc_free_0(v27, v30[0] + 1LL);
  }
  result = sub_CAE810(a1, v18, v19, v20, v21);
  *(_QWORD *)(a1 + 80) = result;
  if ( !result )
  {
LABEL_10:
    *(_BYTE *)(a1 + 77) = 1;
    *(_QWORD *)(a1 + 80) = 0;
  }
LABEL_11:
  v17 = (__int64)v24;
  if ( v24 != v25 )
    return j_j___libc_free_0(v17, v25[0] + 1LL);
  return result;
}
