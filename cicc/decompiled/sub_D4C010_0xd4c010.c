// Function: sub_D4C010
// Address: 0xd4c010
//
_QWORD *__fastcall sub_D4C010(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r13
  _QWORD *result; // rax
  __int64 v6; // rdx
  _QWORD *v7; // rbx
  unsigned __int64 v8; // rax
  __int64 v9; // r15
  int v10; // eax
  unsigned int v11; // ebx
  int v12; // r14d
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r13
  __int64 *v18; // rax
  __int64 *v19; // rdx
  __int64 *v20; // rax
  char v21; // dl
  __int64 v22; // rax
  const void *v23; // [rsp+8h] [rbp-188h]
  _QWORD *v25; // [rsp+20h] [rbp-170h]
  _QWORD *v26; // [rsp+28h] [rbp-168h]
  __int64 v27; // [rsp+30h] [rbp-160h]
  __int64 v28; // [rsp+40h] [rbp-150h] BYREF
  __int64 *v29; // [rsp+48h] [rbp-148h]
  __int64 v30; // [rsp+50h] [rbp-140h]
  int v31; // [rsp+58h] [rbp-138h]
  char v32; // [rsp+5Ch] [rbp-134h]
  char v33; // [rsp+60h] [rbp-130h] BYREF

  v29 = (__int64 *)&v33;
  v3 = *(_QWORD **)(a1 + 40);
  result = *(_QWORD **)(a1 + 32);
  v27 = a2;
  v28 = 0;
  v30 = 32;
  v31 = 0;
  v32 = 1;
  if ( result != v3 )
  {
    while ( 1 )
    {
      v6 = *result;
      v7 = result;
      if ( a3 != *result )
        break;
      if ( v3 == ++result )
        return result;
    }
    if ( v3 != result )
    {
      v23 = (const void *)(a2 + 16);
      while ( 1 )
      {
        v8 = *(_QWORD *)(v6 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v8 == v6 + 48 )
          goto LABEL_19;
        if ( !v8 )
          BUG();
        v9 = v8 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30 > 0xA )
          goto LABEL_19;
        v10 = sub_B46E30(v9);
        if ( !v10 )
          goto LABEL_19;
        v26 = v3;
        v25 = v7;
        v11 = 0;
        v12 = v10;
        do
        {
          while ( 1 )
          {
            a2 = v11;
            v13 = sub_B46EC0(v9, v11);
            v17 = v13;
            if ( *(_BYTE *)(a1 + 84) )
            {
              v18 = *(__int64 **)(a1 + 64);
              v19 = &v18[*(unsigned int *)(a1 + 76)];
              if ( v18 != v19 )
              {
                while ( v17 != *v18 )
                {
                  if ( v19 == ++v18 )
                    goto LABEL_28;
                }
                goto LABEL_17;
              }
            }
            else
            {
              a2 = v13;
              if ( sub_C8CA60(a1 + 56, v13) )
                goto LABEL_17;
            }
LABEL_28:
            if ( v32 )
            {
              v20 = v29;
              v14 = HIDWORD(v30);
              v19 = &v29[HIDWORD(v30)];
              if ( v29 != v19 )
              {
                while ( v17 != *v20 )
                {
                  if ( v19 == ++v20 )
                    goto LABEL_39;
                }
                goto LABEL_17;
              }
LABEL_39:
              if ( HIDWORD(v30) < (unsigned int)v30 )
                break;
            }
            a2 = v17;
            sub_C8CC70((__int64)&v28, v17, (__int64)v19, v14, v15, v16);
            if ( v21 )
              goto LABEL_35;
LABEL_17:
            if ( v12 == ++v11 )
              goto LABEL_18;
          }
          ++HIDWORD(v30);
          *v19 = v17;
          ++v28;
LABEL_35:
          a2 = v27;
          v22 = *(unsigned int *)(v27 + 8);
          if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(v27 + 12) )
          {
            a2 = (__int64)v23;
            sub_C8D5F0(v27, v23, v22 + 1, 8u, v15, v16);
            v22 = *(unsigned int *)(v27 + 8);
          }
          ++v11;
          *(_QWORD *)(*(_QWORD *)v27 + 8 * v22) = v17;
          ++*(_DWORD *)(v27 + 8);
        }
        while ( v12 != v11 );
LABEL_18:
        v3 = v26;
        v7 = v25;
LABEL_19:
        result = v7 + 1;
        if ( v3 != v7 + 1 )
        {
          while ( 1 )
          {
            v6 = *result;
            v7 = result;
            if ( a3 != *result )
              break;
            if ( v3 == ++result )
              goto LABEL_24;
          }
          if ( v3 != result )
            continue;
        }
LABEL_24:
        if ( !v32 )
          return (_QWORD *)_libc_free(v29, a2);
        return result;
      }
    }
  }
  return result;
}
