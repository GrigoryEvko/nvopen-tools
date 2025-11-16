// Function: sub_1EE4FF0
// Address: 0x1ee4ff0
//
__int64 __fastcall sub_1EE4FF0(
        __int64 a1,
        __int64 a2,
        char a3,
        int a4,
        __int64 a5,
        unsigned int a6,
        unsigned __int8 (__fastcall *a7)(__int64, __int64))
{
  __int64 v10; // rdi
  __int64 result; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // r10
  unsigned int v15; // eax
  __int64 v16; // r9
  unsigned int v17; // eax
  __int64 v18; // rcx
  __int64 v19; // r11
  __int64 v20; // r13
  unsigned int v21; // ebx
  __int64 v22; // r11
  __int64 v23; // rdi
  _QWORD *v24; // rsi
  _QWORD *v25; // rdx
  __int64 v26; // [rsp+10h] [rbp-40h]
  unsigned int v27; // [rsp+10h] [rbp-40h]
  __int64 v28; // [rsp+18h] [rbp-38h]
  __int64 v29; // [rsp+18h] [rbp-38h]
  __int64 v30; // [rsp+18h] [rbp-38h]
  __int64 v31; // [rsp+18h] [rbp-38h]

  if ( a4 < 0 )
  {
    v13 = *(unsigned int *)(a1 + 408);
    v14 = a2;
    v15 = a4 & 0x7FFFFFFF;
    v16 = a4 & 0x7FFFFFFF;
    if ( (a4 & 0x7FFFFFFFu) < (unsigned int)v13 )
    {
      v19 = *(_QWORD *)(*(_QWORD *)(a1 + 400) + 8LL * v15);
      if ( v19 )
      {
LABEL_9:
        if ( !a3 )
          return (unsigned int)-a7(v19, a5);
        v20 = *(_QWORD *)(v19 + 104);
        v21 = 0;
        if ( v20 )
        {
          do
          {
            if ( a7(v20, a5) )
              v21 |= *(_DWORD *)(v20 + 112);
            v20 = *(_QWORD *)(v20 + 104);
          }
          while ( v20 );
        }
        else
        {
          v30 = v14;
          if ( a7(v19, a5) )
            return (unsigned int)sub_1E69F40(v30, a4);
        }
        return v21;
      }
    }
    v17 = v15 + 1;
    if ( (unsigned int)v13 < v17 )
    {
      v22 = v17;
      if ( v17 < v13 )
      {
        *(_DWORD *)(a1 + 408) = v17;
      }
      else if ( v17 > v13 )
      {
        if ( v17 > (unsigned __int64)*(unsigned int *)(a1 + 412) )
        {
          v27 = v17;
          v31 = v17;
          sub_16CD150(a1 + 400, (const void *)(a1 + 416), v17, 8, a5, v16);
          v13 = *(unsigned int *)(a1 + 408);
          v16 = a4 & 0x7FFFFFFF;
          v14 = a2;
          v17 = v27;
          v22 = v31;
        }
        v18 = *(_QWORD *)(a1 + 400);
        v23 = *(_QWORD *)(a1 + 416);
        v24 = (_QWORD *)(v18 + 8 * v22);
        v25 = (_QWORD *)(v18 + 8 * v13);
        if ( v24 != v25 )
        {
          do
            *v25++ = v23;
          while ( v24 != v25 );
          v18 = *(_QWORD *)(a1 + 400);
        }
        *(_DWORD *)(a1 + 408) = v17;
        goto LABEL_8;
      }
    }
    v18 = *(_QWORD *)(a1 + 400);
LABEL_8:
    v26 = v14;
    v28 = v16;
    *(_QWORD *)(v18 + 8LL * (a4 & 0x7FFFFFFF)) = sub_1DBA290(a4);
    v29 = *(_QWORD *)(*(_QWORD *)(a1 + 400) + 8 * v28);
    sub_1DBB110((_QWORD *)a1, v29);
    v14 = v26;
    v19 = v29;
    goto LABEL_9;
  }
  v10 = *(_QWORD *)(*(_QWORD *)(a1 + 672) + 8LL * (unsigned int)a4);
  result = a6;
  if ( v10 )
    return (unsigned int)(a7(v10, a5) == 0) - 1;
  return result;
}
