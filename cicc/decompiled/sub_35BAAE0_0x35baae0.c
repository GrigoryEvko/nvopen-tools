// Function: sub_35BAAE0
// Address: 0x35baae0
//
__int64 __fastcall sub_35BAAE0(_QWORD *a1)
{
  __int64 v1; // r14
  __int64 v2; // rax
  __int64 result; // rax
  unsigned int v5; // r12d
  unsigned int v6; // r15d
  _DWORD *v7; // rsi
  __int64 v8; // rax
  int v9; // r9d
  __int64 v10; // rdx
  _DWORD *v11; // rdi
  __int64 v12; // r10
  __int64 v13; // rsi
  __int64 v14; // r10
  _QWORD *v15; // [rsp+8h] [rbp-58h]
  int v16; // [rsp+1Ch] [rbp-44h]
  unsigned int v17; // [rsp+28h] [rbp-38h] BYREF
  int v18[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v1 = *a1;
  v2 = *(_QWORD *)(*a1 + 168LL) - *(_QWORD *)(*a1 + 160LL);
  v18[0] = 0;
  result = 0xAAAAAAAAAAAAAAABLL * (v2 >> 5);
  if ( (_DWORD)result )
  {
    v5 = result;
    do
    {
      v7 = *(_DWORD **)(v1 + 192);
      if ( v7 == sub_35B8320(*(_DWORD **)(v1 + 184), (__int64)v7, v18) )
      {
        v6 = v18[0];
        v16 = -1431655765 * ((__int64)(*(_QWORD *)(v1 + 168) - *(_QWORD *)(v1 + 160)) >> 5);
        goto LABEL_6;
      }
      v6 = v18[0] + 1;
      v18[0] = v6;
    }
    while ( v5 > v6 );
    v16 = -1431655765 * ((__int64)(*(_QWORD *)(v1 + 168) - *(_QWORD *)(v1 + 160)) >> 5);
LABEL_6:
    result = (__int64)(a1 + 13);
    v15 = a1 + 13;
LABEL_7:
    while ( v16 != v6 )
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)(*a1 + 160LL) + 96LL * v6;
        v9 = *(_DWORD *)(v8 + 16);
        if ( *(_QWORD *)(v8 + 80) - *(_QWORD *)(v8 + 72) <= 8u )
        {
          v17 = v6;
          v14 = (__int64)(a1 + 1);
          v18[0] = v6;
          switch ( v9 )
          {
            case 2:
              sub_35B9090(a1 + 7, (unsigned int *)v18);
              v14 = (__int64)(a1 + 1);
              break;
            case 3:
              sub_35B9090(a1 + 1, (unsigned int *)v18);
              v14 = (__int64)(a1 + 1);
              break;
            case 1:
              sub_35B9090(v15, (unsigned int *)v18);
              v14 = (__int64)(a1 + 1);
              break;
          }
          sub_B99820(v14, &v17);
          result = *(_QWORD *)(*a1 + 160LL) + 96LL * v17;
          *(_DWORD *)(result + 16) = 3;
        }
        else
        {
          v10 = *(unsigned int *)(v8 + 20);
          if ( *(_DWORD *)(v8 + 24) >= (unsigned int)v10
            && (v18[0] = 0, v11 = *(_DWORD **)(v8 + 32), &v11[v10] == sub_35B8490(v11, (__int64)&v11[v10], v18)) )
          {
            v17 = v6;
            v18[0] = v6;
            switch ( v9 )
            {
              case 2:
                sub_35B9090(a1 + 7, (unsigned int *)v18);
                break;
              case 3:
                sub_35B9090(a1 + 1, (unsigned int *)v18);
                break;
              case 1:
                sub_35B9090(v15, (unsigned int *)v18);
                break;
            }
            sub_B99820((__int64)v15, &v17);
            result = *(_QWORD *)(*a1 + 160LL) + 96LL * v17;
            *(_DWORD *)(result + 16) = 1;
          }
          else
          {
            v17 = v6;
            v12 = (__int64)(a1 + 7);
            v18[0] = v6;
            switch ( v9 )
            {
              case 2:
                sub_35B9090(a1 + 7, (unsigned int *)v18);
                v12 = (__int64)(a1 + 7);
                break;
              case 3:
                sub_35B9090(a1 + 1, (unsigned int *)v18);
                v12 = (__int64)(a1 + 7);
                break;
              case 1:
                sub_35B9090(v15, (unsigned int *)v18);
                v12 = (__int64)(a1 + 7);
                break;
            }
            sub_B99820(v12, &v17);
            result = *(_QWORD *)(*a1 + 160LL) + 96LL * v17;
            *(_DWORD *)(result + 16) = 2;
            *(_BYTE *)(result + 64) = 1;
          }
        }
        v18[0] = ++v6;
        if ( v5 <= v6 )
          break;
        while ( 1 )
        {
          v13 = *(_QWORD *)(v1 + 192);
          result = (__int64)sub_35B8320(*(_DWORD **)(v1 + 184), v13, v18);
          if ( v13 == result )
            break;
          result = (unsigned int)v18[0];
          v6 = v18[0] + 1;
          v18[0] = v6;
          if ( v5 <= v6 )
            goto LABEL_7;
        }
        v6 = v18[0];
        if ( v16 == v18[0] )
          return result;
      }
    }
  }
  return result;
}
