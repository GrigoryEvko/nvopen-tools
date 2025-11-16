// Function: sub_C57680
// Address: 0xc57680
//
__int64 __fastcall sub_C57680(
        __int64 a1,
        const char *a2,
        __int64 *a3,
        _QWORD *a4,
        _DWORD *a5,
        _BYTE *a6,
        __int64 **a7,
        __int64 *a8)
{
  size_t v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  bool v15; // zf
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  int v19; // eax
  __int64 v20; // rsi
  __int64 result; // rax
  __int64 v22; // rcx
  _QWORD *v23; // rdx
  __int64 *v24; // r12
  __int64 *v25; // r13
  char v26; // di
  __int64 v27; // rsi
  __int64 v28; // rcx
  _QWORD *v29; // rdx
  const char *v32; // [rsp+20h] [rbp-60h] BYREF
  char v33; // [rsp+40h] [rbp-40h]
  char v34; // [rsp+41h] [rbp-3Fh]

  v11 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v11);
  v15 = *(_QWORD *)(a1 + 136) == 0;
  v16 = *a3;
  v17 = a3[1];
  *(_QWORD *)(a1 + 40) = *a3;
  *(_QWORD *)(a1 + 48) = v17;
  if ( v15 )
  {
    *(_QWORD *)(a1 + 136) = *a4;
  }
  else
  {
    v18 = sub_CEADF0(a1, a2, v16, v12, v13, v14);
    v34 = 1;
    v32 = "cl::location(x) specified more than once!";
    v33 = 3;
    sub_C53280(a1, (__int64)&v32, 0, 0, v18);
  }
  v19 = (32 * (*a5 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  *(_BYTE *)(a1 + 12) = v19;
  *(_BYTE *)(a1 + 12) = (8 * (*a6 & 3)) | v19 & 0xE7;
  sub_C57500(a1, *a7);
  v20 = *a8;
  if ( *a8 )
  {
    if ( !*(_BYTE *)(a1 + 124) )
      return sub_C8CC70(a1 + 96, v20);
    result = *(_QWORD *)(a1 + 104);
    v22 = *(unsigned int *)(a1 + 116);
    v23 = (_QWORD *)(result + 8 * v22);
    if ( (_QWORD *)result == v23 )
    {
LABEL_26:
      if ( (unsigned int)v22 < *(_DWORD *)(a1 + 112) )
      {
        *(_DWORD *)(a1 + 116) = v22 + 1;
        *v23 = v20;
        ++*(_QWORD *)(a1 + 96);
        return result;
      }
      return sub_C8CC70(a1 + 96, v20);
    }
    while ( v20 != *(_QWORD *)result )
    {
      result += 8;
      if ( v23 == (_QWORD *)result )
        goto LABEL_26;
    }
  }
  else
  {
    result = a8[1];
    if ( result )
    {
      v24 = *(__int64 **)result;
      result = *(unsigned int *)(result + 8);
      v25 = &v24[result];
      if ( v24 != v25 )
      {
        v26 = *(_BYTE *)(a1 + 124);
        v27 = *v24;
        if ( !v26 )
          goto LABEL_20;
LABEL_14:
        result = *(_QWORD *)(a1 + 104);
        v28 = *(unsigned int *)(a1 + 116);
        v29 = (_QWORD *)(result + 8 * v28);
        if ( (_QWORD *)result == v29 )
        {
LABEL_22:
          if ( (unsigned int)v28 >= *(_DWORD *)(a1 + 112) )
          {
LABEL_20:
            while ( 1 )
            {
              ++v24;
              result = sub_C8CC70(a1 + 96, v27);
              v26 = *(_BYTE *)(a1 + 124);
              if ( v25 == v24 )
                break;
LABEL_19:
              v27 = *v24;
              if ( v26 )
                goto LABEL_14;
            }
          }
          else
          {
            ++v24;
            *(_DWORD *)(a1 + 116) = v28 + 1;
            *v29 = v27;
            v26 = *(_BYTE *)(a1 + 124);
            ++*(_QWORD *)(a1 + 96);
            if ( v25 != v24 )
              goto LABEL_19;
          }
        }
        else
        {
          while ( v27 != *(_QWORD *)result )
          {
            result += 8;
            if ( v29 == (_QWORD *)result )
              goto LABEL_22;
          }
          if ( v25 != ++v24 )
            goto LABEL_19;
        }
      }
    }
  }
  return result;
}
