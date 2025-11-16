// Function: sub_8E72C0
// Address: 0x8e72c0
//
unsigned __int8 *__fastcall sub_8E72C0(unsigned __int8 *a1, int a2, __int64 a3)
{
  int v4; // r13d
  unsigned __int8 *v5; // rax
  unsigned __int8 *v6; // r12
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rcx
  unsigned __int8 v11; // bl
  int v12; // eax
  char v13; // dl
  char *v14; // rcx
  unsigned __int8 *v15; // rax
  char v16; // dl
  char *v17; // rcx
  __int64 v18[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = a2;
  v5 = sub_8E5810(a1, v18, a3);
  v6 = v5;
  if ( v18[0] > 0 )
  {
    if ( a2 )
    {
LABEL_3:
      v6 += v18[0];
      return v6;
    }
    if ( v18[0] > 8 )
    {
      v13 = 95;
      v14 = "INTERNAL";
      while ( *v5++ == v13 )
      {
        v13 = *v14++;
        if ( !v13 )
          goto LABEL_3;
      }
    }
    v15 = v6;
    v16 = 95;
    v17 = "GLOBAL__N_";
    if ( v18[0] <= 10 )
    {
LABEL_30:
      v4 = 1;
      goto LABEL_13;
    }
    do
    {
      if ( *v15++ != v16 )
        goto LABEL_30;
      v16 = *v17++;
    }
    while ( v16 );
    if ( !*(_QWORD *)(a3 + 32) )
    {
      sub_8E5790("_NV_ANON_NAMESPACE", a3);
      if ( v18[0] <= 0 )
        return v6;
    }
LABEL_13:
    while ( 1 )
    {
      v11 = *v6;
      if ( !*v6 )
        break;
      v12 = isalnum(v11);
      if ( v11 == 36 || v11 == 95 || v12 )
      {
        if ( v4 && !*(_QWORD *)(a3 + 32) )
        {
          v8 = *(_QWORD *)(a3 + 8);
          v9 = v8 + 1;
          if ( !*(_DWORD *)(a3 + 28) )
          {
            v10 = *(_QWORD *)(a3 + 16);
            if ( v9 < v10 )
            {
              *(_BYTE *)(*(_QWORD *)a3 + v8) = v11;
              v9 = *(_QWORD *)(a3 + 8) + 1LL;
            }
            else
            {
              *(_DWORD *)(a3 + 28) = 1;
              if ( v10 )
              {
                *(_BYTE *)(*(_QWORD *)a3 + v10 - 1) = 0;
                v9 = *(_QWORD *)(a3 + 8) + 1LL;
              }
            }
          }
          *(_QWORD *)(a3 + 8) = v9;
        }
      }
      else if ( v4 )
      {
        break;
      }
      ++v6;
      if ( --v18[0] <= 0 )
        return v6;
    }
  }
  if ( *(_DWORD *)(a3 + 24) )
    return v6;
  ++*(_QWORD *)(a3 + 32);
  ++*(_QWORD *)(a3 + 48);
  *(_DWORD *)(a3 + 24) = 1;
  return v6;
}
