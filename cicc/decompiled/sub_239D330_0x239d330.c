// Function: sub_239D330
// Address: 0x239d330
//
__int64 __fastcall sub_239D330(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  __int64 *v11; // rbx
  __int64 *v12; // r12
  __int64 v13; // rdx
  __int64 result; // rax
  __int64 v15; // r12
  __int64 *v16; // rax
  __int64 v17; // r12
  __int64 *v18; // r14
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // r12
  __int64 v21; // rdx
  __int64 v22; // r13
  __int64 *v23; // r14
  __int64 *src; // [rsp+8h] [rbp-78h]
  __int64 v25; // [rsp+10h] [rbp-70h] BYREF
  __int64 v26; // [rsp+18h] [rbp-68h]
  __int64 v27; // [rsp+20h] [rbp-60h]
  __int64 v28; // [rsp+30h] [rbp-50h] BYREF
  __int64 v29; // [rsp+38h] [rbp-48h]
  __int64 v30; // [rsp+40h] [rbp-40h]

  v9 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v10 = (*a1 >> 2) & 1;
  if ( (_DWORD)v10 )
  {
    v15 = *(unsigned int *)(v9 + 8);
    v11 = *(__int64 **)v9;
    v25 = a7;
    v26 = a8;
    v15 *= 8;
    v27 = a9;
    v16 = (__int64 *)((char *)v11 + v15);
    v28 = a7;
    v10 = v15 >> 3;
    v17 = v15 >> 5;
    src = v16;
    v29 = a8;
    v30 = a9;
    if ( v17 )
    {
      while ( 1 )
      {
        if ( (unsigned __int8)sub_239C850((__int64)&v28, *v11) )
        {
          v12 = src;
          goto LABEL_20;
        }
        if ( (unsigned __int8)sub_239C850((__int64)&v28, v11[1]) )
        {
          v12 = src;
          ++v11;
          goto LABEL_20;
        }
        if ( (unsigned __int8)sub_239C850((__int64)&v28, v11[2]) )
        {
          v12 = src;
          v11 += 2;
          goto LABEL_20;
        }
        if ( (unsigned __int8)sub_239C850((__int64)&v28, v11[3]) )
          break;
        v11 += 4;
        if ( !--v17 )
        {
          v10 = src - v11;
          goto LABEL_35;
        }
      }
      v12 = src;
      v11 += 3;
      goto LABEL_20;
    }
LABEL_35:
    if ( v10 == 2 )
    {
      v23 = src;
LABEL_37:
      if ( (unsigned __int8)sub_239C850((__int64)&v28, *v11) )
      {
        v12 = src;
        src = v23;
        goto LABEL_20;
      }
      v12 = src;
      src = v23;
      ++v11;
      goto LABEL_39;
    }
    v12 = src;
  }
  else
  {
    v11 = a1;
    if ( v9 )
    {
      v10 = 1;
      src = a1 + 1;
      v12 = a1 + 1;
    }
    else
    {
      src = a1;
      v12 = a1;
    }
    v25 = a7;
    v26 = a8;
    v27 = a9;
    v28 = a7;
    v29 = a8;
    v30 = a9;
  }
  if ( v10 == 3 )
  {
    if ( (unsigned __int8)sub_239C850((__int64)&v28, *v11) )
      goto LABEL_20;
    v23 = src;
    ++v11;
    src = v12;
    goto LABEL_37;
  }
  if ( v10 != 1 )
  {
LABEL_7:
    v11 = v12;
    goto LABEL_8;
  }
LABEL_39:
  if ( !(unsigned __int8)sub_239C850((__int64)&v28, *v11) )
    goto LABEL_7;
LABEL_20:
  if ( v11 != v12 )
  {
    v18 = v11 + 1;
    if ( v11 + 1 != v12 )
    {
      do
      {
        if ( !(unsigned __int8)sub_239C850((__int64)&v25, *v18) )
          *v11++ = *v18;
        ++v18;
      }
      while ( v18 != v12 );
      v13 = *a1;
      result = (*a1 >> 2) & 1;
      if ( ((*a1 >> 2) & 1) == 0 )
        goto LABEL_9;
      goto LABEL_26;
    }
  }
LABEL_8:
  v13 = *a1;
  result = (*a1 >> 2) & 1;
  if ( ((*a1 >> 2) & 1) == 0 )
  {
LABEL_9:
    if ( v11 != src && a1 == v11 )
      *a1 = 0;
    return result;
  }
LABEL_26:
  if ( v13 )
  {
    if ( (_BYTE)result )
    {
      v19 = v13 & 0xFFFFFFFFFFFFFFF8LL;
      v20 = v19;
      if ( v19 )
      {
        result = *(_QWORD *)v19;
        v21 = *(_QWORD *)v19 + 8LL * *(unsigned int *)(v19 + 8);
        v22 = v21 - (_QWORD)src;
        if ( (__int64 *)v21 != src )
        {
          memmove(v11, src, v21 - (_QWORD)src);
          result = *(_QWORD *)v20;
        }
        *(_DWORD *)(v20 + 8) = ((__int64)v11 + v22 - result) >> 3;
      }
    }
  }
  return result;
}
