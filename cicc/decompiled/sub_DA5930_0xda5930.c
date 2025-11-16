// Function: sub_DA5930
// Address: 0xda5930
//
unsigned __int64 __fastcall sub_DA5930(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 result; // rax
  unsigned __int64 *v5; // r14
  _QWORD *v6; // r8
  unsigned __int64 *v7; // r9
  __int64 v8; // r12
  __int64 v9; // r15
  char *v10; // rax
  char *v11; // r13
  unsigned int v12; // r9d
  unsigned int v13; // r11d
  unsigned int v14; // edi
  unsigned __int64 *v15; // rcx
  __int64 v16; // rax
  unsigned __int64 v17; // r8
  unsigned int v18; // eax
  __int16 v19; // r10
  unsigned __int64 *v20; // rsi
  unsigned __int64 *v21; // rdx
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // rdx
  _QWORD **v24; // [rsp+8h] [rbp-A8h]
  unsigned __int64 *v25; // [rsp+10h] [rbp-A0h]
  __int64 v26; // [rsp+18h] [rbp-98h] BYREF
  __int64 v27; // [rsp+28h] [rbp-88h]
  _QWORD v28[4]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v29; // [rsp+50h] [rbp-60h] BYREF
  int v30; // [rsp+58h] [rbp-58h] BYREF
  __int64 v31; // [rsp+60h] [rbp-50h]
  int *v32; // [rsp+68h] [rbp-48h]
  int *v33; // [rsp+70h] [rbp-40h]
  __int64 v34; // [rsp+78h] [rbp-38h]

  result = *(unsigned int *)(a1 + 8);
  v26 = a2;
  if ( result > 1 )
  {
    v30 = 0;
    v32 = &v30;
    v5 = *(unsigned __int64 **)a1;
    v33 = &v30;
    v31 = 0;
    v34 = 0;
    v28[0] = &v29;
    v28[1] = &v26;
    v28[2] = a3;
    if ( result == 2 )
    {
      v27 = sub_DA4700(&v29, a2, v5[1], *v5, a3, 0);
      if ( BYTE4(v27) )
      {
        if ( (int)v27 < 0 )
        {
          v23 = v5[1];
          v5[1] = *v5;
          *v5 = v23;
        }
      }
    }
    else
    {
      v6 = v28;
      v7 = &v5[result];
      v8 = (__int64)(8 * result) >> 3;
      do
      {
        v9 = 8 * v8;
        v24 = (_QWORD **)v6;
        v25 = v7;
        v10 = (char *)sub_2207800(8 * v8, &unk_435FF63);
        v7 = v25;
        v6 = v24;
        v11 = v10;
        if ( v10 )
        {
          sub_DA5830(v5, v25, v10, (void *)v8, v24);
          goto LABEL_6;
        }
        v8 >>= 1;
      }
      while ( v8 );
      v9 = 0;
      sub_DA51B0(v5, v25, v24);
LABEL_6:
      j_j___libc_free_0(v11, v9);
      v12 = *(_DWORD *)(a1 + 8);
      v13 = v12 - 2;
      if ( v12 != 2 )
      {
        v14 = 0;
        do
        {
          while ( 1 )
          {
            v15 = *(unsigned __int64 **)a1;
            v16 = v14++;
            v17 = *(_QWORD *)(*(_QWORD *)a1 + 8 * v16);
            v18 = v14;
            v19 = *(_WORD *)(v17 + 24);
            if ( v14 != v12 )
              break;
            v14 = v12;
            if ( v13 == v12 )
              return sub_D91E90(v31);
          }
          while ( 1 )
          {
            v20 = &v15[v18];
            if ( v19 != *(_WORD *)(*v20 + 24) )
              break;
            if ( *v20 == v17 )
            {
              v21 = &v15[v14];
              v22 = *v21;
              *v21 = v17;
              *v20 = v22;
              if ( v13 == v14 )
                return sub_D91E90(v31);
              ++v18;
              ++v14;
              if ( v18 == v12 )
                break;
            }
            else if ( ++v18 == v12 )
            {
              break;
            }
            v15 = *(unsigned __int64 **)a1;
          }
        }
        while ( v13 != v14 );
      }
    }
    return sub_D91E90(v31);
  }
  return result;
}
