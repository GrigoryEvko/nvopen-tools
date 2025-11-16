// Function: sub_2A5D020
// Address: 0x2a5d020
//
void __fastcall sub_2A5D020(__int64 a1, unsigned __int64 a2, __int64 *a3)
{
  __int64 v4; // rcx
  __int64 v5; // rdx
  unsigned __int64 *v7; // rax
  unsigned int v8; // ecx
  unsigned __int64 *i; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 j; // rbx
  unsigned __int64 v14; // rax
  bool v15; // zf
  char v16; // cl
  __int64 v17; // rdx
  __int64 *v18; // rsi
  __int64 v19; // rdi
  unsigned __int64 *v20; // rdi
  __int64 v21; // rdx
  unsigned __int64 v22; // [rsp+8h] [rbp-98h] BYREF
  unsigned __int64 v23; // [rsp+18h] [rbp-88h] BYREF
  __int64 v24[2]; // [rsp+20h] [rbp-80h] BYREF
  unsigned __int64 v25; // [rsp+30h] [rbp-70h]
  unsigned __int64 v26; // [rsp+38h] [rbp-68h]
  __int64 v27; // [rsp+40h] [rbp-60h]
  unsigned __int64 *v28; // [rsp+48h] [rbp-58h]
  unsigned __int64 *v29; // [rsp+50h] [rbp-50h]
  __int64 v30; // [rsp+58h] [rbp-48h]
  __int64 v31; // [rsp+60h] [rbp-40h]
  __int64 v32; // [rsp+68h] [rbp-38h]

  v4 = *a3;
  v22 = a2;
  v5 = *(_QWORD *)(v4 + 8LL * ((unsigned int)a2 >> 6));
  if ( !_bittest64(&v5, a2) )
  {
    v24[0] = 0;
    v24[1] = 0;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    v28 = 0;
    v29 = 0;
    v30 = 0;
    v31 = 0;
    v32 = 0;
    sub_2A5CBD0(v24, 0);
    v7 = v29;
    if ( v29 == (unsigned __int64 *)(v31 - 8) )
    {
      sub_FE0450(v24, &v22);
      v8 = v22;
    }
    else
    {
      v8 = v22;
      if ( v29 )
      {
        *v29 = v22;
        v7 = v29;
      }
      v29 = v7 + 1;
    }
    *(_QWORD *)(*a3 + 8LL * (v8 >> 6)) |= 1LL << v8;
    for ( i = (unsigned __int64 *)v25; v29 != (unsigned __int64 *)v25; i = (unsigned __int64 *)v25 )
    {
      v10 = *i;
      v22 = *i;
      if ( i == (unsigned __int64 *)(v27 - 8) )
      {
        j_j___libc_free_0(v26);
        v21 = *++v28 + 512;
        v26 = *v28;
        v27 = v21;
        v10 = v22;
        v25 = v26;
      }
      else
      {
        v25 = (unsigned __int64)(i + 1);
      }
      v11 = **(_QWORD **)(a1 + 8) + 80 * v10;
      v12 = *(_QWORD *)(v11 + 40);
      for ( j = *(_QWORD *)(v11 + 32); v12 != j; *v18 |= 1LL << v16 )
      {
        while ( 1 )
        {
          v14 = *(_QWORD *)(*(_QWORD *)j + 8LL);
          v15 = *(_QWORD *)(*(_QWORD *)j + 32LL) == 0;
          v23 = v14;
          if ( !v15 )
          {
            v16 = v14 & 0x3F;
            v17 = 8LL * ((unsigned int)v14 >> 6);
            v18 = (__int64 *)(v17 + *a3);
            v19 = *v18;
            if ( !_bittest64(&v19, v14) )
              break;
          }
          j += 8;
          if ( v12 == j )
            goto LABEL_19;
        }
        v20 = v29;
        if ( v29 == (unsigned __int64 *)(v31 - 8) )
        {
          sub_FE0450(v24, &v23);
          v16 = v23 & 0x3F;
          v18 = (__int64 *)(*a3 + 8LL * ((unsigned int)v23 >> 6));
        }
        else
        {
          if ( v29 )
          {
            *v29 = v14;
            v20 = v29;
            v18 = (__int64 *)(v17 + *a3);
          }
          v29 = v20 + 1;
        }
        j += 8;
      }
LABEL_19:
      ;
    }
    sub_2A5BF70((unsigned __int64 *)v24);
  }
}
