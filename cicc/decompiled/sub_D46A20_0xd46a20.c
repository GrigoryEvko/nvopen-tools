// Function: sub_D46A20
// Address: 0xd46a20
//
__int64 __fastcall sub_D46A20(__int64 a1, __int64 a2)
{
  __int64 v3; // rcx
  __int64 result; // rax
  __int64 v5; // rdi
  __int64 *v6; // rcx
  __int64 *v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // r13
  int v10; // r12d
  unsigned int v11; // r15d
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r14
  __int64 *v17; // rax
  __int64 *v18; // rdx
  __int64 *v19; // rax
  char v20; // dl
  __int64 v21; // rax
  __int64 *v22; // [rsp+0h] [rbp-180h]
  __int64 v23; // [rsp+8h] [rbp-178h]
  __int64 v25; // [rsp+20h] [rbp-160h]
  __int64 v26; // [rsp+28h] [rbp-158h]
  __int64 v27; // [rsp+30h] [rbp-150h] BYREF
  __int64 *v28; // [rsp+38h] [rbp-148h]
  __int64 v29; // [rsp+40h] [rbp-140h]
  int v30; // [rsp+48h] [rbp-138h]
  char v31; // [rsp+4Ch] [rbp-134h]
  char v32; // [rsp+50h] [rbp-130h] BYREF

  v3 = a1 + 56;
  v28 = (__int64 *)&v32;
  result = *(_QWORD *)(a1 + 40);
  v5 = *(_QWORD *)(a1 + 32);
  v25 = v3;
  v6 = (__int64 *)(a2 + 16);
  v7 = &v27;
  v27 = 0;
  v29 = 32;
  v30 = 0;
  v31 = 1;
  v23 = result;
  v26 = v5;
  v22 = v6;
  if ( v5 != result )
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(*(_QWORD *)v26 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v8 != *(_QWORD *)v26 + 48LL )
      {
        if ( !v8 )
          BUG();
        v9 = v8 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30 <= 0xA )
        {
          v10 = sub_B46E30(v9);
          if ( v10 )
            break;
        }
      }
LABEL_13:
      v26 += 8;
      result = v26;
      if ( v23 == v26 )
      {
        if ( !v31 )
          return _libc_free(v28, v7);
        return result;
      }
    }
    v11 = 0;
    while ( 1 )
    {
      v7 = (__int64 *)v11;
      v12 = sub_B46EC0(v9, v11);
      v16 = v12;
      if ( *(_BYTE *)(a1 + 84) )
      {
        v17 = *(__int64 **)(a1 + 64);
        v18 = &v17[*(unsigned int *)(a1 + 76)];
        if ( v17 != v18 )
        {
          while ( v16 != *v17 )
          {
            if ( v18 == ++v17 )
              goto LABEL_18;
          }
          goto LABEL_12;
        }
      }
      else
      {
        v7 = (__int64 *)v12;
        if ( sub_C8CA60(v25, v12) )
          goto LABEL_12;
      }
LABEL_18:
      if ( v31 )
      {
        v19 = v28;
        v13 = HIDWORD(v29);
        v18 = &v28[HIDWORD(v29)];
        if ( v28 != v18 )
        {
          while ( v16 != *v19 )
          {
            if ( v18 == ++v19 )
              goto LABEL_29;
          }
          goto LABEL_12;
        }
LABEL_29:
        if ( HIDWORD(v29) < (unsigned int)v29 )
        {
          ++HIDWORD(v29);
          *v18 = v16;
          ++v27;
          goto LABEL_25;
        }
      }
      v7 = (__int64 *)v16;
      sub_C8CC70((__int64)&v27, v16, (__int64)v18, v13, v14, v15);
      if ( v20 )
      {
LABEL_25:
        v7 = (__int64 *)a2;
        v21 = *(unsigned int *)(a2 + 8);
        if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          v7 = v22;
          sub_C8D5F0(a2, v22, v21 + 1, 8u, v14, v15);
          v21 = *(unsigned int *)(a2 + 8);
        }
        ++v11;
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v21) = v16;
        ++*(_DWORD *)(a2 + 8);
        if ( v10 == v11 )
          goto LABEL_13;
      }
      else
      {
LABEL_12:
        if ( v10 == ++v11 )
          goto LABEL_13;
      }
    }
  }
  return result;
}
