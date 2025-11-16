// Function: sub_2AD8BC0
// Address: 0x2ad8bc0
//
__int64 **__fastcall sub_2AD8BC0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 **result; // rax
  __int64 *v6; // rdx
  __int64 v7; // r14
  char v8; // dl
  __int64 v9; // r15
  __int64 v10; // r9
  __int64 *v11; // rdx
  __int64 v12; // rdi
  __int64 *v13; // rax
  _QWORD *v14; // rax
  unsigned __int64 v15; // rdi
  _QWORD *v16; // rdx
  int v17; // r14d
  int v18; // eax
  __int64 v19; // [rsp+8h] [rbp-48h]
  unsigned __int64 v20[7]; // [rsp+18h] [rbp-38h] BYREF

  v1 = a1 + 112;
  v3 = *(_QWORD *)(a1 + 96);
  v4 = *(unsigned int *)(a1 + 104);
  while ( 1 )
  {
    result = (__int64 **)(v3 + 24 * v4 - 24);
    v6 = result[1];
    if ( v6 == *result )
      break;
    result[1] = v6 + 1;
    v7 = *v6;
    sub_AE6EC0(a1, *v6);
    v4 = *(unsigned int *)(a1 + 104);
    if ( v8 )
    {
      v9 = *(_QWORD *)(v7 + 80);
      v10 = v9 + 8LL * *(unsigned int *)(v7 + 88);
      if ( *(_DWORD *)(a1 + 108) <= (unsigned int)v4 )
      {
        v19 = v9 + 8LL * *(unsigned int *)(v7 + 88);
        v3 = sub_C8D7D0(a1 + 96, v1, 0, 0x18u, v20, v10);
        v12 = 3LL * *(unsigned int *)(a1 + 104);
        v13 = (__int64 *)(v12 * 8 + v3);
        if ( v12 * 8 + v3 )
        {
          v13[1] = v9;
          v13[2] = v7;
          *v13 = v19;
          v12 = 3LL * *(unsigned int *)(a1 + 104);
        }
        v14 = *(_QWORD **)(a1 + 96);
        v15 = (unsigned __int64)&v14[v12];
        if ( v14 != (_QWORD *)v15 )
        {
          v16 = (_QWORD *)v3;
          do
          {
            if ( v16 )
            {
              *v16 = *v14;
              v16[1] = v14[1];
              v16[2] = v14[2];
            }
            v14 += 3;
            v16 += 3;
          }
          while ( (_QWORD *)v15 != v14 );
          v15 = *(_QWORD *)(a1 + 96);
        }
        v17 = v20[0];
        if ( v1 != v15 )
          _libc_free(v15);
        v18 = *(_DWORD *)(a1 + 104);
        *(_QWORD *)(a1 + 96) = v3;
        *(_DWORD *)(a1 + 108) = v17;
        v4 = (unsigned int)(v18 + 1);
        *(_DWORD *)(a1 + 104) = v4;
      }
      else
      {
        v3 = *(_QWORD *)(a1 + 96);
        v11 = (__int64 *)(v3 + 24LL * (unsigned int)v4);
        if ( v11 )
        {
          *v11 = v10;
          v11[1] = v9;
          v11[2] = v7;
          LODWORD(v4) = *(_DWORD *)(a1 + 104);
          v3 = *(_QWORD *)(a1 + 96);
        }
        v4 = (unsigned int)(v4 + 1);
        *(_DWORD *)(a1 + 104) = v4;
      }
    }
    else
    {
      v3 = *(_QWORD *)(a1 + 96);
    }
  }
  return result;
}
